/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Numerical contract admission, strict values and design-only tensor geometry. No card is opened. */
#include <stdio.h>
#include <string.h>

#include "anemone/backend_seam.h"
#include "anemone/startup_policy.h"

#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
#define INVALID_ENUMERATOR 99
#define TARGET_HIDDEN 5120u
#define TARGET_HEADS 64u
#define TARGET_EXPERT_WIDTH 2304u
#define TARGET_LAYER_COUNT 40u
#define TP4_LAYER_COUNT 29u
#define EXPECTED_REDUCTIONS 2u
#define TP5_CMP_HIDDEN 896u
#define TP5_PRO_HIDDEN 1536u
#define TP5_CMP_HEADS 11u
#define TP5_PRO_HEADS 20u
#define TP5_CMP_EXPERT 432u
#define TP5_PRO_EXPERT 576u
#define TP4_HIDDEN 1280u
#define TP4_HEADS 16u
#define TP4_PAIR_RANKS 2u
#define TP4_EXPERT_SMALL 512u
#define TP4_EXPERT_LARGE 640u

static int failures;
_Static_assert(
	EXECUTION_TP4_EXL3 == EXECUTION_TP2_SINGLE_SINGLE_PRO + 1,
	"EXL3 placement is appended without renumbering existing placements");

_Static_assert(sizeof(enum numerics_code) == sizeof(uint32_t),
	       "the malformed numerical word fits");
_Static_assert(sizeof(enum execution_placement_code) == sizeof(uint32_t),
	       "the malformed placement word fits");
_Static_assert(sizeof(enum cmp_order_code) == sizeof(uint32_t),
	       "the malformed CMP order word fits");

static bool policies_equal(const struct startup_policy *first,
			   const struct startup_policy *second)
{
	return first->numerics == second->numerics &&
	       first->placement == second->placement &&
	       first->cmp_order == second->cmp_order;
}

/* Compare semantic fields: structure padding is outside the prepare contract. */
static bool designs_equal(const struct tensor_placement_design *first,
			  const struct tensor_placement_design *second)
{
	if (first->rank_count != second->rank_count ||
	    first->pro_card != second->pro_card ||
	    first->tensor_layer_count != second->tensor_layer_count ||
	    first->target_layer_count != second->target_layer_count ||
	    first->reductions_per_layer != second->reductions_per_layer ||
	    first->kernel_aligned != second->kernel_aligned) {
		return false;
	}
	for (uint32_t rank = 0; rank < TENSOR_PLACEMENT_CARDS; rank++) {
		const struct tensor_shard_geometry *const first_shard =
			&first->ranks[rank];
		const struct tensor_shard_geometry *const second_shard =
			&second->ranks[rank];
		if (first_shard->card != second_shard->card ||
		    first_shard->hidden_begin != second_shard->hidden_begin ||
		    first_shard->hidden_count != second_shard->hidden_count ||
		    first_shard->attention_head_begin !=
			    second_shard->attention_head_begin ||
		    first_shard->attention_head_count !=
			    second_shard->attention_head_count ||
		    first_shard->expert_intermediate_begin !=
			    second_shard->expert_intermediate_begin ||
		    first_shard->expert_intermediate_count !=
			    second_shard->expert_intermediate_count) {
			return false;
		}
	}
	return true;
}

static void expect(bool condition, const char *what)
{
	if (!condition) {
		(void)fprintf(stderr, "FAILED: %s\n", what);
		failures++;
	}
}

static void test_text(void)
{
	static const struct {
		const char *text;
		enum numerics_code numerics;
	} numerical_values[] = { { "exact", NUMERICS_EXACT },
				 { "relaxed", NUMERICS_RELAXED },
				 { "fast", NUMERICS_FAST } };
	static const struct {
		const char *text;
		enum execution_placement_code placement;
	} placement_values[] = { { "pp5", EXECUTION_PP5 },
				 { "tp5", EXECUTION_TP5 },
				 { "tp4_pro", EXECUTION_TP4_PRO },
				 { "tp4_exl3", EXECUTION_TP4_EXL3 },
				 { "tp2_pp2_pro", EXECUTION_TP2_PP2_PRO },
				 { "tp2_single_single_pro",
				   EXECUTION_TP2_SINGLE_SINGLE_PRO } };
	static const char *const invalid_values[] = {
		"",	    " ",	   " exact",	  "exact ",
		"EXACT",    "relax",	   "fast=1",	  "pp",
		"tp4",	    "TP5",	   "tp5 ",	  "tp4_pro\n",
		"tp2",	    "tp2xpp2_pro", "TP2_PP2_PRO", "tp2_pp2_pro ",
		"TP4_EXL3", "tp4_exl3 "
	};
	for (uint32_t index = 0; index < COUNT_OF(numerical_values); index++) {
		enum numerics_code numerical = NUMERICS_FAST;
		expect(numerics_from_text(numerical_values[index].text,
					  &numerical) == STATUS_OK &&
			       numerical == numerical_values[index].numerics,
		       "exact numerical values parse");
	}
	for (uint32_t index = 0; index < COUNT_OF(placement_values); index++) {
		enum execution_placement_code placement = EXECUTION_TP5;
		expect(execution_placement_from_text(
			       placement_values[index].text, &placement) ==
				       STATUS_OK &&
			       placement == placement_values[index].placement,
		       "exact placement values parse");
	}
	for (uint32_t index = 0; index < COUNT_OF(invalid_values); index++) {
		enum numerics_code numerical = NUMERICS_FAST;
		enum execution_placement_code placement = EXECUTION_TP5;
		expect(numerics_from_text(invalid_values[index], &numerical) ==
				       STATUS_INVALID_ARGUMENT &&
			       numerical == NUMERICS_FAST,
		       "invalid numerical values leave output untouched");
		expect(execution_placement_from_text(invalid_values[index],
						     &placement) ==
				       STATUS_INVALID_ARGUMENT &&
			       placement == EXECUTION_TP5,
		       "invalid placement values leave output untouched");
	}
	enum numerics_code numerical = NUMERICS_FAST;
	enum execution_placement_code placement = EXECUTION_TP5;
	expect(numerics_from_text(NULL, &numerical) ==
			       STATUS_INVALID_ARGUMENT &&
		       numerical == NUMERICS_FAST &&
		       numerics_from_text("exact", NULL) ==
			       STATUS_INVALID_ARGUMENT &&
		       execution_placement_from_text(NULL, &placement) ==
			       STATUS_INVALID_ARGUMENT &&
		       placement == EXECUTION_TP5 &&
		       execution_placement_from_text("pp5", NULL) ==
			       STATUS_INVALID_ARGUMENT,
	       "NULL text arguments refuse");
}

static void test_policy(void)
{
	static const struct {
		struct startup_policy_request request;
		result_status status;
		enum numerics_code numerical;
		enum cmp_order_code cmp;
	} cases[] = {
		{ { 0 }, STATUS_OK, NUMERICS_EXACT, CMP_ORDER_REFERENCE },
		{ { .numerics = NUMERICS_EXACT, .numerics_was_given = true },
		  STATUS_OK,
		  NUMERICS_EXACT,
		  CMP_ORDER_REFERENCE },
		{ { .cmp_order = CMP_ORDER_RELAXED,
		    .cmp_order_was_given = true },
		  STATUS_OK,
		  NUMERICS_RELAXED,
		  CMP_ORDER_RELAXED },
		{ { .numerics = NUMERICS_RELAXED, .numerics_was_given = true },
		  STATUS_OK,
		  NUMERICS_RELAXED,
		  CMP_ORDER_RELAXED },
		{ { .numerics = NUMERICS_FAST, .numerics_was_given = true },
		  STATUS_OK,
		  NUMERICS_FAST,
		  CMP_ORDER_RELAXED },
		{ { .numerics = NUMERICS_FAST,
		    .numerics_was_given = true,
		    .cmp_order_was_given = true },
		  STATUS_OK,
		  NUMERICS_FAST,
		  CMP_ORDER_REFERENCE },
		{ { .numerics = NUMERICS_EXACT,
		    .numerics_was_given = true,
		    .cmp_order = CMP_ORDER_RELAXED,
		    .cmp_order_was_given = true },
		  STATUS_INVALID_ARGUMENT,
		  0,
		  0 },
		{ { .placement = EXECUTION_TP5 }, STATUS_UNSUPPORTED, 0, 0 },
		{ { .placement = EXECUTION_TP4_PRO },
		  STATUS_OK,
		  NUMERICS_EXACT,
		  CMP_ORDER_REFERENCE },
		{ { .placement = EXECUTION_TP5,
		    .numerics = NUMERICS_RELAXED,
		    .numerics_was_given = true },
		  STATUS_OK,
		  NUMERICS_RELAXED,
		  CMP_ORDER_RELAXED },
		{ { .placement = EXECUTION_TP4_PRO,
		    .numerics = NUMERICS_FAST,
		    .numerics_was_given = true },
		  STATUS_OK,
		  NUMERICS_FAST,
		  CMP_ORDER_REFERENCE },
		{ { .placement = EXECUTION_TP2_SINGLE_SINGLE_PRO },
		  STATUS_UNSUPPORTED,
		  0,
		  0 },
		{ { .placement = EXECUTION_TP2_SINGLE_SINGLE_PRO,
		    .numerics = NUMERICS_RELAXED,
		    .numerics_was_given = true },
		  STATUS_OK,
		  NUMERICS_RELAXED,
		  CMP_ORDER_REFERENCE },
		{ { .placement = EXECUTION_TP4_PRO,
		    .numerics = NUMERICS_RELAXED,
		    .numerics_was_given = true },
		  STATUS_OK,
		  NUMERICS_RELAXED,
		  CMP_ORDER_REFERENCE },
		{ { .placement = EXECUTION_TP2_PP2_PRO },
		  STATUS_UNSUPPORTED,
		  0,
		  0 },
		{ { .placement = EXECUTION_TP2_PP2_PRO,
		    .numerics = NUMERICS_RELAXED,
		    .numerics_was_given = true },
		  STATUS_OK,
		  NUMERICS_RELAXED,
		  CMP_ORDER_REFERENCE },
		{ { .placement = EXECUTION_TP2_PP2_PRO,
		    .numerics = NUMERICS_FAST,
		    .numerics_was_given = true },
		  STATUS_OK,
		  NUMERICS_FAST,
		  CMP_ORDER_REFERENCE },
		{ { .placement = EXECUTION_TP2_PP2_PRO,
		    .cmp_order = CMP_ORDER_RELAXED,
		    .cmp_order_was_given = true },
		  STATUS_OK,
		  NUMERICS_RELAXED,
		  CMP_ORDER_RELAXED },
		{ { .placement = EXECUTION_TP2_PP2_PRO,
		    .numerics = NUMERICS_FAST,
		    .numerics_was_given = true,
		    .cmp_order_was_given = true },
		  STATUS_OK,
		  NUMERICS_FAST,
		  CMP_ORDER_REFERENCE },
		{ { .placement = EXECUTION_TP2_PP2_PRO,
		    .numerics = NUMERICS_EXACT,
		    .numerics_was_given = true,
		    .cmp_order = CMP_ORDER_RELAXED,
		    .cmp_order_was_given = true },
		  STATUS_INVALID_ARGUMENT,
		  0,
		  0 },
		{ { .placement = EXECUTION_TP4_EXL3 },
		  STATUS_OK,
		  NUMERICS_EXACT,
		  CMP_ORDER_REFERENCE },
		{ { .placement = EXECUTION_TP4_EXL3,
		    .numerics = NUMERICS_RELAXED,
		    .numerics_was_given = true },
		  STATUS_OK,
		  NUMERICS_RELAXED,
		  CMP_ORDER_REFERENCE },
		{ { .placement = EXECUTION_TP4_EXL3,
		    .numerics = NUMERICS_FAST,
		    .numerics_was_given = true },
		  STATUS_OK,
		  NUMERICS_FAST,
		  CMP_ORDER_REFERENCE },
		{ { .placement = EXECUTION_TP4_EXL3,
		    .numerics = NUMERICS_EXACT,
		    .numerics_was_given = true,
		    .cmp_order = CMP_ORDER_RELAXED,
		    .cmp_order_was_given = true },
		  STATUS_INVALID_ARGUMENT,
		  0,
		  0 },
		{ { .placement = EXECUTION_TP4_EXL3,
		    .numerics = NUMERICS_RELAXED,
		    .numerics_was_given = true,
		    .cmp_order = CMP_ORDER_RELAXED,
		    .cmp_order_was_given = true },
		  STATUS_OK,
		  NUMERICS_RELAXED,
		  CMP_ORDER_RELAXED },
	};
	for (uint32_t index = 0; index < COUNT_OF(cases); index++) {
		struct startup_policy policy = { .numerics = NUMERICS_FAST,
						 .placement = EXECUTION_TP5,
						 .cmp_order =
							 CMP_ORDER_RELAXED };
		const struct startup_policy before = policy;
		const result_status status =
			startup_policy_prepare(&cases[index].request, &policy);
		expect(status == cases[index].status,
		       "policy admission follows the numerical contract");
		if (status == STATUS_OK) {
			expect(policy.numerics == cases[index].numerical &&
				       policy.cmp_order == cases[index].cmp &&
				       policy.placement ==
					       cases[index].request.placement,
			       "policy retains selected arithmetic and placement");
			const bool tensor_pair =
				(policy.placement == EXECUTION_TP2_PP2_PRO ||
				 policy.placement == EXECUTION_TP4_PRO ||
				 policy.placement == EXECUTION_TP4_EXL3 ||
				 policy.placement ==
					 EXECUTION_TP2_SINGLE_SINGLE_PRO) &&
				(policy.numerics != NUMERICS_EXACT ||
				 policy.placement == EXECUTION_TP4_PRO ||
				 policy.placement == EXECUTION_TP4_EXL3) &&
				policy.cmp_order == CMP_ORDER_REFERENCE;
			expect(startup_policy_admit(&policy) ==
				       (policy.placement == EXECUTION_PP5 ||
							tensor_pair ?
						STATUS_OK :
						STATUS_UNSUPPORTED),
			       "only implemented execution is admitted");
		} else {
			expect(policies_equal(&policy, &before),
			       "refused policy is untouched");
		}
	}
	struct startup_policy_request request = { 0 };
	struct startup_policy policy = { 0 }, before = policy;
	expect(startup_policy_prepare(NULL, &policy) ==
			       STATUS_INVALID_ARGUMENT &&
		       policies_equal(&policy, &before) &&
		       startup_policy_prepare(&request, NULL) ==
			       STATUS_INVALID_ARGUMENT &&
		       startup_policy_admit(NULL) == STATUS_INVALID_ARGUMENT,
	       "NULL policy arguments refuse");
	policy.cmp_order = CMP_ORDER_RELAXED;
	expect(startup_policy_admit(&policy) == STATUS_INVALID_ARGUMENT,
	       "runtime admission rechecks manually constructed contradictory contracts");
	const uint32_t invalid_word = INVALID_ENUMERATOR;
	memcpy(&policy.numerics, &invalid_word, sizeof(policy.numerics));
	expect(startup_policy_admit(&policy) == STATUS_INVALID_ARGUMENT,
	       "runtime admission rechecks enumerators");
	static const uint32_t malformed_words[] = { INVALID_ENUMERATOR,
						    UINT32_MAX };
	for (uint32_t index = 0; index < COUNT_OF(malformed_words); index++) {
		request = (struct startup_policy_request){ 0 };
		policy = (struct startup_policy){ 0 };
		before = policy;
		/* Exercise malformed C11 representations without an out-of-range enumeration conversion. */
		memcpy(&request.numerics, &malformed_words[index],
		       sizeof(request.numerics));
		expect(startup_policy_prepare(&request, &policy) ==
				       STATUS_INVALID_ARGUMENT &&
			       policies_equal(&policy, &before),
		       "malformed numerical word refuses without mutation");
		request = (struct startup_policy_request){ 0 };
		memcpy(&request.placement, &malformed_words[index],
		       sizeof(request.placement));
		expect(startup_policy_prepare(&request, &policy) ==
				       STATUS_INVALID_ARGUMENT &&
			       policies_equal(&policy, &before),
		       "malformed placement word refuses without mutation");
		request = (struct startup_policy_request){ 0 };
		memcpy(&request.cmp_order, &malformed_words[index],
		       sizeof(request.cmp_order));
		expect(startup_policy_prepare(&request, &policy) ==
				       STATUS_INVALID_ARGUMENT &&
			       policies_equal(&policy, &before),
		       "malformed CMP word refuses without mutation");
	}
}

static void test_tensor_design(void)
{
	for (uint32_t pro_card = 0; pro_card < TENSOR_PLACEMENT_CARDS;
	     pro_card++) {
		uint32_t classes[TENSOR_PLACEMENT_CARDS];
		for (uint32_t card = 0; card < TENSOR_PLACEMENT_CARDS; card++) {
			classes[card] = card == pro_card ?
						ARITHMETIC_CLASS_SM_120_NATIVE :
						ARITHMETIC_CLASS_SM_80;
		}
		for (uint32_t mode = EXECUTION_TP5; mode <= EXECUTION_TP4_PRO;
		     mode++) {
			const struct startup_policy policy = {
				.numerics = NUMERICS_RELAXED,
				.placement =
					(enum execution_placement_code)mode,
				.cmp_order = CMP_ORDER_RELAXED
			};
			struct tensor_placement_design design = { 0 };
			const bool uneven = policy.placement == EXECUTION_TP5;
			expect(tensor_placement_design_prepare(
				       &policy, classes, COUNT_OF(classes),
				       &design) == STATUS_OK,
			       "design prepares for every PRO ordinal");
			const uint32_t expected_ranks =
				uneven ? TENSOR_PLACEMENT_CARDS :
					 TENSOR_PLACEMENT_CMP_CARDS;
			expect(design.rank_count == expected_ranks &&
				       design.pro_card == pro_card &&
				       design.target_layer_count ==
					       TARGET_LAYER_COUNT &&
				       design.tensor_layer_count ==
					       (uneven ? TARGET_LAYER_COUNT :
							 TP4_LAYER_COUNT) &&
				       design.reductions_per_layer ==
					       EXPECTED_REDUCTIONS &&
				       design.kernel_aligned != uneven,
			       "design retains dependent stages and exposes incompatible TP5 kernel dimensions");
			uint32_t hidden = 0, heads = 0, expert = 0,
				 card_mask = 0;
			for (uint32_t rank = 0; rank < expected_ranks; rank++) {
				const struct tensor_shard_geometry *const shard =
					&design.ranks[rank];
				const bool is_pro = shard->card == pro_card;
				expect(shard->card < TENSOR_PLACEMENT_CARDS &&
					       (card_mask &
						(1u << shard->card)) == 0 &&
					       shard->hidden_begin == hidden &&
					       shard->attention_head_begin ==
						       heads &&
					       shard->expert_intermediate_begin ==
						       expert,
				       "tensor ownership is unique and ranges are contiguous");
				expect(shard->hidden_count ==
						       (uneven ?
								is_pro ?
								TP5_PRO_HIDDEN :
								TP5_CMP_HIDDEN :
								TP4_HIDDEN) &&
					       shard->attention_head_count ==
						       (uneven ? is_pro ?
								 TP5_PRO_HEADS :
								 TP5_CMP_HEADS :
								 TP4_HEADS) &&
					       shard->expert_intermediate_count ==
						       (uneven ?
								is_pro ?
								TP5_PRO_EXPERT :
								TP5_CMP_EXPERT :
								(rank % TP4_PAIR_RANKS ==
										 0 ?
									 TP4_EXPERT_SMALL :
									 TP4_EXPERT_LARGE)),
				       "design preserves the modeled uneven shares");
				card_mask |= 1u << shard->card;
				hidden += shard->hidden_count;
				heads += shard->attention_head_count;
				expert += shard->expert_intermediate_count;
			}
			expect(hidden == TARGET_HIDDEN &&
				       heads == TARGET_HEADS &&
				       expert == TARGET_EXPERT_WIDTH,
			       "design conserves the original tensor dimensions exactly");
			expect(startup_policy_admit(&policy) ==
				       STATUS_UNSUPPORTED,
			       "geometry never grants execution admission");
		}
	}
	uint32_t classes[TENSOR_PLACEMENT_CARDS] = { ARITHMETIC_CLASS_SM_120,
						     ARITHMETIC_CLASS_SM_80,
						     ARITHMETIC_CLASS_SM_80,
						     ARITHMETIC_CLASS_SM_80,
						     ARITHMETIC_CLASS_SM_80 };
	struct startup_policy policy = { .numerics = NUMERICS_FAST,
					 .placement = EXECUTION_TP5,
					 .cmp_order = CMP_ORDER_RELAXED };
	struct tensor_placement_design design = {
		.rank_count = INVALID_ENUMERATOR,
		.pro_card = INVALID_ENUMERATOR,
		.tensor_layer_count = INVALID_ENUMERATOR,
		.target_layer_count = INVALID_ENUMERATOR,
		.reductions_per_layer = INVALID_ENUMERATOR,
		.kernel_aligned = true
	};
	for (uint32_t rank = 0; rank < TENSOR_PLACEMENT_CARDS; rank++) {
		design.ranks[rank] = (struct tensor_shard_geometry){
			.card = INVALID_ENUMERATOR,
			.hidden_begin = INVALID_ENUMERATOR,
			.hidden_count = INVALID_ENUMERATOR,
			.attention_head_begin = INVALID_ENUMERATOR,
			.attention_head_count = INVALID_ENUMERATOR,
			.expert_intermediate_begin = INVALID_ENUMERATOR,
			.expert_intermediate_count = INVALID_ENUMERATOR
		};
	}
	const struct tensor_placement_design before = design;
	expect(tensor_placement_design_prepare(NULL, classes, COUNT_OF(classes),
					       &design) ==
			       STATUS_INVALID_ARGUMENT &&
		       tensor_placement_design_prepare(
			       &policy, NULL, COUNT_OF(classes), &design) ==
			       STATUS_INVALID_ARGUMENT &&
		       tensor_placement_design_prepare(
			       &policy, classes, COUNT_OF(classes), NULL) ==
			       STATUS_INVALID_ARGUMENT &&
		       tensor_placement_design_prepare(
			       &policy, classes, COUNT_OF(classes) - 1u,
			       &design) == STATUS_INVALID_ARGUMENT &&
		       designs_equal(&design, &before),
	       "bad tensor arguments leave output untouched");
	classes[1] = ARITHMETIC_CLASS_SM_120;
	expect(tensor_placement_design_prepare(&policy, classes,
					       COUNT_OF(classes),
					       &design) == STATUS_UNSUPPORTED &&
		       designs_equal(&design, &before),
	       "two PRO cards cannot masquerade as the design hardware");
	classes[1] = ARITHMETIC_CLASS_SM_80;
	policy.placement = EXECUTION_TP2_PP2_PRO;
	expect(tensor_placement_design_prepare(&policy, classes,
					       COUNT_OF(classes),
					       &design) == STATUS_UNSUPPORTED &&
		       designs_equal(&design, &before),
	       "two TP2 pipeline groups never masquerade as one TP4 geometry");
	const uint32_t cmp_classes[TENSOR_PLACEMENT_CMP_CARDS] = {
		ARITHMETIC_CLASS_SM_80, ARITHMETIC_CLASS_SM_80,
		ARITHMETIC_CLASS_SM_80, ARITHMETIC_CLASS_SM_80
	};
	const struct startup_policy exl3 = { .numerics = NUMERICS_EXACT,
					     .placement = EXECUTION_TP4_EXL3,
					     .cmp_order = CMP_ORDER_REFERENCE };
	expect(tensor_placement_design_prepare(&exl3, cmp_classes,
					       COUNT_OF(cmp_classes),
					       &design) == STATUS_UNSUPPORTED &&
		       designs_equal(&design, &before),
	       "four-CMP EXL3 execution policy does not invent a tensor design geometry");
	policy.numerics = NUMERICS_EXACT;
	policy.cmp_order = CMP_ORDER_REFERENCE;
	expect(tensor_placement_design_prepare(&policy, classes,
					       COUNT_OF(classes),
					       &design) == STATUS_UNSUPPORTED &&
		       designs_equal(&design, &before),
	       "modeled exact TP does not acquire equivalence");
}

int main(void)
{
	test_text();
	test_policy();
	test_tensor_design();
	(void)printf("startup numerical/placement policy: %s\n",
		     failures == 0 ? "PASS" : "FAIL");
	return failures == 0 ? 0 : 1;
}
