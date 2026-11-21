/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_attention_core_card.h: what the card test of the attention core (ds41f_attention_core_card_test.c) runs its
 * synthetic cases and its replay with: a case's arrays as the host holds them and the reference's output of each row in
 * each kernel's form, a launch plan, and one launch of a plan on a card in either order. Test support: no part of the
 * engine.
 *
 * Contract:
 *   - A case (struct ds41f_test_attention_case) is a step of up to 8 rows: its step rows, sinks and the arrays below,
 *     and ds41f_test_attention_case_hold holds them for the counts its caller set (session_count, pool_ratio,
 *     pool_entries): queries zero, ring and pool DS41F_TEST_ATTENTION_POISON (an FP16 not-a-number: where no row may
 *     read), selection -1, rotation and the references unset. False behind a failure line;
 *     ds41f_test_attention_case_free frees what it held, also of a case it did not finish, and zeroes the case.
 *   - ds41f_test_attention_reference_row is the host reference's input of row `row` of a case
 *     (attention_core_reference.h): its queries, the keys its window and its selection name (NULL for none), the sinks,
 *     its rotation use; every head, in the kernel form of the order. ds41f_test_attention_case_references computes
 *     reference[order] of every row for both orders: false behind a failure line.
 *   - A plan (struct ds41f_test_attention_plan) is one launch: rows source_rows[0 ..] of the case as its rows 0 ..,
 *     program_rows of them prepared, step_rows of them in its step; from step_rows on, the selections name
 *     DS41F_TEST_ATTENTION_FAR_ENTRY where far_selections_behind (no launch may read them). The two breaks of the test:
 *     window_one_later puts each step row's position one later on the card than the case's where the context
 *     allows; largest_word sets one FP16 word of the newest window row of the case's row 0 to 65504 on the card,
 *     where the plan's row 0 is the case's.
 *   - ds41f_test_attention_plan_run lays the plan out on a card, prepares its launch in the order (with block_window,
 *     the draft's block core: window_mode DS41F_ATTENTION_CORE_WINDOW_BLOCK), runs it and fetches its output into
 *     output_out [program_rows][64][512]: what the launch only reads must come back as it went, and no launch may
 *     write a matrix (the sinks lie in the weights region). False behind a failure line.
 *   - ds41f_test_attention_is_untouched: every byte of count values is DS41F_TEST_ATTENTION_UNTOUCHED_OUTPUT, what the
 *     output holds before a launch.
 */
#ifndef DS41F_ATTENTION_CORE_CARD_H
#define DS41F_ATTENTION_CORE_CARD_H

#include <stdbool.h>
#include <stdint.h>

#include "models/deepseek_v41_flash/attention/ds41f_attention_core.h"
#include "ds41f_attention_core_reference.h"
#include "engine/library/block_scaled_linear/linear_card_fixture.h"

#define DS41F_TEST_ATTENTION_POISON 0x7E00u           /* an FP16 not-a-number */
#define DS41F_TEST_ATTENTION_UNTOUCHED_OUTPUT 0x5A    /* the output's bytes before a launch */
#define DS41F_TEST_ATTENTION_FAR_ENTRY 0x7FFFFFF0     /* a selection's entry beyond every pool */
#define DS41F_TEST_ATTENTION_ORDERS 2u /* enum linear_order_code: fixed, reference */
#define DS41F_TEST_ATTENTION_ROTATION_FLOATS \
	(DS41F_ROTATION_ROW_BYTES / sizeof(float))
#define DS41F_TEST_ATTENTION_USE_FLOATS \
	(DS41F_TEST_ATTENTION_ROTATION_FLOATS / DS41F_ROTATION_USES)

struct ds41f_test_attention_case {
    const char *name;
    uint32_t session_count;
    uint32_t context_token_limit;
    uint32_t pool_ratio;
    uint32_t rotation_use;                      /* the prepare's: theta 1 with a pool (REF:680-687) */
    bool exact;                                 /* every value the reference's bits */
    uint64_t pool_entries;                      /* of a session */
    struct step_row_record step_rows[LINEAR_MAXIMUM_ROWS];
    float sinks[DS41F_HEADS_VALUE];
    uint16_t *queries;                          /* [8][64][512] */
    uint16_t *ring;                             /* [sessions][256][512] */
    uint16_t *pool;                             /* [sessions][entries][512]; NULL without a pool */
    int32_t *selection;                         /* [8][512] */
    float *rotation;                            /* [8][3][32][2] */
    double *reference[DS41F_TEST_ATTENTION_ORDERS];   /* [8][64][512] each: the kernel form of each order */
};

struct ds41f_test_attention_plan {
    const char *name;
    uint32_t source_rows[LINEAR_MAXIMUM_ROWS];
    uint32_t program_rows;
    uint32_t step_rows;
    uint32_t first_head;
    uint32_t head_count;
    bool far_selections_behind;
    bool window_one_later;
    bool largest_word;
    bool block_window;                          /* the draft's block core (ds41f_attention_core_records.h) */
};

bool ds41f_test_attention_case_hold(struct ds41f_test_attention_case *data);
void ds41f_test_attention_case_free(struct ds41f_test_attention_case *data);
void ds41f_test_attention_reference_row(const struct ds41f_test_attention_case *data,
				  uint32_t row, enum linear_order_code order,
				  struct ds41f_test_attention_row *reference);
bool ds41f_test_attention_case_references(struct ds41f_test_attention_case *data);
/* Rows first_row .. first_row + row_count - 1 of a case, every head, every row in the step, no break. */
void ds41f_test_attention_plan_of_rows(struct ds41f_test_attention_plan *plan, const char *name, uint32_t first_row,
                                 uint32_t row_count);
bool ds41f_test_attention_plan_run(struct test_linear_card *card,
			     const struct ds41f_attention_core_kernels *kernels,
			     const struct ds41f_test_attention_case *data,
			     const struct ds41f_test_attention_plan *plan,
			     enum linear_order_code order,
			     uint16_t *output_out);
bool ds41f_test_attention_is_untouched(const uint16_t *values, uint64_t count);

#endif
