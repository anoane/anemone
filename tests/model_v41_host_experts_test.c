/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * model_v41_host_experts_test: DeepSeek-V4.1-Flash with routed experts in host memory (source/model_v41.h,
 * model_v41_program.c, model_seam.h's expert pauses), on packs of the model's names and shapes held as tables only
 * (model_v41_pack.h) and the backend in host memory, against the same placement with every expert on the cards.
 *
 *   - The five cards of first_token_plan.md 1.3, with the last HOST_EXPERTS experts of layer HOST_LAYER (card 0) and
 *     every expert of layer ALL_HOST_LAYER (the last card) in host memory: those two cards begin their state with
 *     MODEL_EXPERTS_PER_ROW times ROWS slots of an expert's bytes, aligned, and hold the rest of the state
 *     of the other placement behind them; the other cards hold its state exactly. The image's tables name an expert
 *     in host memory by its tag, every other where the layout puts it; the two layers' experts' range runs to the
 *     end of the slots; the plan holds each host expert's offset and its matrices' place; state_byte_count gives
 *     each card's state and, for host memory, the bytes of those experts.
 *   - The programs of 1 and ROWS rows: each of the two layers in two segments on its card, the first ending with
 *     its router and an expert pause of the layer's job table, the card's slots and its experts in host memory; the
 *     router's record takes the tags; the launches are the other placement's; the step driver takes the programs,
 *     and refuses copies of them that lay the n-gram rows, or a hand-off's source or destination, over the slots.
 *   - Refusals of an expert in host memory: with a matrix apart from the others, with one on the card, with its
 *     matrices together but in another order than the others' (the layout moved), and beyond the host arena.
 * A failure is one line on the standard output: the case, and what was expected of it.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/engram.h"
#include "log_expectation.h"
#include "anemone/hooks.h"
#include "anemone/loader.h"
#include "anemone/router_records.h"
#include "anemone/site_front_records.h"
#include "anemone/step.h"
#include "engram_pack.h"
#include "host_memory_backend.h"
#include "memory_storage.h"
#include "model_v41.h"
#include "model_v41_pack.h"

#define CARDS TEST_MODEL_V41_CARDS
#define PRO_CARD (CARDS - 1u)
#define UNIT_COUNT TEST_MODEL_V41_UNIT_COUNT
#define EXPERTS MODEL_EXPERTS
#define MATRICES MODEL_EXPERT_MATRICES
#define CONTEXT 16384u
#define ROWS 8u
#define SESSIONS 1u
#define HOST_LAYER 5u                              /* on card 0 */
#define HOST_EXPERTS 24u                           /* its last */
#define ALL_HOST_LAYER 33u                         /* on the last card: all of its experts */
#define NEIGHBOUR_EXPERT 382u                      /* of HOST_LAYER, in host memory beside expert 383 */
#define PAUSED_LAYERS 2u
#define HOST_CARD_BYTES 4096u                      /* an arena no launch reads: the programs only name offsets */
#define NAME_CAPACITY 64
#define CAPTURE_CAPACITY 4096u
#define PROGRAMS 2u                                /* of 1 and ROWS rows */
#define WAIT_LIMIT 1000000000ull
#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))

static int failures = 0;
static const char *context_text = "";
static char capture_path[] = "/tmp/anemone_model_v41_host_experts_test_XXXXXX";
static int capture_descriptor = -1;
static int error_stream_descriptor = -1;
static const uint32_t paused_layers[PAUSED_LAYERS] = {HOST_LAYER, ALL_HOST_LAYER};
static const uint32_t first_host_expert[PAUSED_LAYERS] = {EXPERTS - HOST_EXPERTS, 0};
static const char *const matrix_names[MATRICES] = {"w1.weight", "w3.weight", "w2.weight"};   /* gate, up, down */
static const struct model_limits limits = { CONTEXT,
					    ROWS,
					    SESSIONS,
					    STATE_WIDTH_16,
					    LINEAR_PROFILE_PRECISION,
					    LINEAR_ORDER_FIXED,
					    CMP_ORDER_REFERENCE,
					    0,
					    0,
					    0 };
/* What a refusal case changes in the layout: nothing; the extent of the last expert's down and up moved to end where
 * its gate (moved into the extent before) begins; that extent moved behind the host arena. */
enum layout_change { LAYOUT_SAME, LAYOUT_REORDERED, LAYOUT_BEYOND };

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)printf("FAILED: %s: %s\n", context_text, what);
        failures++;
    }
}

static void expect_status(result_status status, result_status expected,
			  const char *what)
{
    if (status != expected) {
	    (void)printf("FAILED: %s: %s: %s, expected %s\n", context_text,
			 what, status_text(status), status_text(expected));
	    failures++;
    }
}

static void capture_begin(void)
{
    expect(capture_descriptor >= 0 && fflush(stderr) == 0 && ftruncate(capture_descriptor, 0) == 0 &&
               lseek(capture_descriptor, 0, SEEK_SET) == 0 && dup2(capture_descriptor, STDERR_FILENO) == STDERR_FILENO,
           "the error stream is captured");
}

/* One error line was logged since capture_begin, and it holds words. */
static void capture_end_expecting(const char *words)
{
    char text[CAPTURE_CAPACITY];
    const bool is_open = capture_descriptor >= 0 && error_stream_descriptor >= 0 && fflush(stderr) == 0;
    const ssize_t byte_count = is_open ? pread(capture_descriptor, text, sizeof(text) - 1, 0) : -1;
    expect(is_open && dup2(error_stream_descriptor, STDERR_FILENO) == STDERR_FILENO && byte_count >= 0,
           "the error stream is restored");
    text[byte_count > 0 ? (size_t)byte_count : 0] = '\0';
    if (!test_log_is_expected(text, words, true)) {
        (void)printf("FAILED: %s: expected one error line with \"%s\", got \"%s\"\n", context_text, words, text);
        failures++;
    }
}

/* ---- placements and plans ---- */

/* The placement of 1.3, with or without the experts of the header comment in host memory. */
static void placement_of(uint32_t *card_of_unit, bool has_host_experts)
{
    test_model_v41_five_cards(card_of_unit, UNIT_COUNT, 0);
    for (uint32_t index = 0; index < PAUSED_LAYERS && has_host_experts; index++) {
        for (uint32_t expert = first_host_expert[index]; expert < EXPERTS; expert++) {
		card_of_unit[TEST_MODEL_V41_EXPERT_UNIT(paused_layers[index],
							expert)] =
			PLACEMENT_IN_HOST_MEMORY;
	}
    }
}

static struct model_plan_record *plan_of(const struct pack_record *pack,
					 bool has_host_experts,
					 struct load_layout **layout_out)
{
    static uint32_t card_of_unit[UNIT_COUNT];
    struct model_plan_record *plan = NULL;
    placement_of(card_of_unit, has_host_experts);
    result_status status = load_layout_create(pack, card_of_unit, UNIT_COUNT,
					      CARDS, layout_out);
    status = status == STATUS_OK ?
		     v41_plan_create(pack, *layout_out, &limits, &plan) :
		     status;
    expect_status(status, STATUS_OK, "the plan is made");
    return plan;
}

/* Matrix m (gate, up, down) of an expert in the pack, or NULL. */
static const struct pack_tensor *matrix_of(const struct pack_record *pack,
					   uint32_t layer, uint32_t expert,
					   uint32_t matrix)
{
    char name[NAME_CAPACITY];
    const int length = snprintf(name, sizeof(name), "layers.%" PRIu32 ".ffn.experts.%" PRIu32 ".%s", layer, expert,
                                matrix_names[matrix]);
    const uint32_t index =
	    length > 0 ? pack_find_tensor(pack, name, (uint32_t)length) :
			 PACK_NOT_FOUND;
    expect(index != PACK_NOT_FOUND, "the pack holds the expert's matrix");
    return index == PACK_NOT_FOUND ? NULL : &pack_tensors_of(pack)[index];
}

/* Where the layout puts matrix m (gate, up, down) of an expert, and its bytes. */
static uint64_t matrix_offset_of(const struct pack_record *pack,
				 const struct load_layout *layout,
				 uint32_t layer, uint32_t expert,
				 uint32_t matrix, uint64_t *byte_count_out)
{
	const struct pack_tensor *const tensor =
		matrix_of(pack, layer, expert, matrix);
	*byte_count_out = tensor != NULL ? tensor->byte_count : 0;
	return tensor != NULL ?
		       layout->byte_offset_of_extent[tensor->extent_index] +
			       tensor->byte_offset_in_extent :
		       0;
}

/* The first byte of an expert's matrices and the end of its last. */
static uint64_t expert_span_of(const struct pack_record *pack,
			       const struct load_layout *layout, uint32_t layer,
			       uint32_t expert, uint64_t *end_out)
{
    uint64_t low = UINT64_MAX, high = 0;
    for (uint32_t matrix = 0; matrix < MATRICES; matrix++) {
        uint64_t byte_count = 0;
        const uint64_t offset = matrix_offset_of(pack, layout, layer, expert, matrix, &byte_count);
        low = offset < low ? offset : low;
        high = offset + byte_count > high ? offset + byte_count : high;
    }
    *end_out = high;
    return low;
}

/* A plan with the experts in host memory against the plan without. */
static void expect_plan(const struct pack_record *pack,
			const struct load_layout *layout,
			const struct model_plan_record *plan,
			const struct model_plan_record *base)
{
    uint64_t end = 0;
    const uint64_t low = expert_span_of(pack, layout, HOST_LAYER, EXPERTS - 1u, &end);
    const uint64_t expert_bytes = end - low;
    const uint64_t slot_bytes =
	    (expert_bytes + V41_DATA_ARRAY_ALIGNMENT_BYTES - 1u) /
	    V41_DATA_ARRAY_ALIGNMENT_BYTES * V41_DATA_ARRAY_ALIGNMENT_BYTES;
    expect(plan->expert_byte_count == expert_bytes && plan->slot_byte_count == slot_bytes,
           "an expert's bytes are its three matrices', a slot those aligned");
    for (uint32_t matrix = 0; matrix < MATRICES; matrix++) {
        uint64_t byte_count = 0;
        expect(plan->expert_matrix_offsets[matrix] ==
                   matrix_offset_of(pack, layout, HOST_LAYER, EXPERTS - 1u, matrix, &byte_count) - low,
               "the plan holds where gate, up and down lie among an expert's bytes");
    }
    for (uint32_t card = 0; card < CARDS; card++) {
	    const struct v41_card *const planned = &plan->cards[card],
					 *const other = &base->cards[card];
	    const bool has_slots = card == plan->layer_card[HOST_LAYER] ||
				   card == plan->layer_card[ALL_HOST_LAYER];
	    const uint64_t slots = has_slots ? (uint64_t)MODEL_EXPERTS_PER_ROW *
						       ROWS * slot_bytes :
					       0;
	    /* each card's weights end where its layout's do: by the experts no longer on it */
	    const uint64_t shift =
		    planned->state_offset + slots - other->state_offset;
	    bool is_shifted =
		    planned->state_offset == layout->card_byte_count[card] &&
		    planned->arena_byte_count ==
			    other->arena_byte_count + shift &&
		    planned->image_byte_count == other->image_byte_count;
	    for (uint32_t array = 0; array < V41_DATA_ARRAY_COUNT; array++) {
		    is_shifted =
			    is_shifted &&
			    (other->array_offset[array] == PROGRAM_NOWHERE ?
				     planned->array_offset[array] ==
					     PROGRAM_NOWHERE :
				     planned->array_offset[array] ==
					     other->array_offset[array] +
						     shift);
	    }
	    for (uint32_t layer = 0; layer < MODEL_LAYERS; layer++) {
		    is_shifted =
			    is_shifted &&
			    (other->ring_offset[layer] == PROGRAM_NOWHERE ?
				     planned->ring_offset[layer] ==
					     PROGRAM_NOWHERE :
				     planned->ring_offset[layer] ==
					     other->ring_offset[layer] +
						     shift) &&
			    (other->pool_offset[layer] == PROGRAM_NOWHERE ?
				     planned->pool_offset[layer] ==
					     PROGRAM_NOWHERE :
				     planned->pool_offset[layer] ==
					     other->pool_offset[layer] + shift);
	    }
	    expect(planned->slot_count ==
				   (has_slots ? MODEL_EXPERTS_PER_ROW * ROWS :
						0) &&
			   planned->slots_byte_count == slots,
		   "the cards of layers with host experts hold the slots of a step's jobs, the others none");
	    expect(is_shifted,
		   "behind the slots, the state of the plan without host experts");
    }
    for (uint32_t layer = 0; layer < MODEL_LAYERS; layer++) {
	    const uint32_t index = layer == HOST_LAYER	   ? 0 :
				   layer == ALL_HOST_LAYER ? 1u :
							     PAUSED_LAYERS;
	    const uint32_t first =
		    index < PAUSED_LAYERS ? first_host_expert[index] : EXPERTS;
	    const struct v41_card *const card =
		    &plan->cards[plan->layer_card[layer]];
	    const unsigned char *const table =
		    card->image + (card->expert_table_offset[layer] -
				   card->state_offset - card->slots_byte_count);
	    bool is_listed = plan->host_expert_count[layer] == EXPERTS - first;
	    for (uint32_t expert = 0; expert < EXPERTS && is_listed; expert++) {
		    uint64_t entries[MATRICES];
		    memcpy(entries, table + (uint64_t)expert * sizeof(entries),
			   sizeof(entries));
		    for (uint32_t matrix = 0; matrix < MATRICES; matrix++) {
			    uint64_t byte_count = 0;
			    const uint64_t offset = matrix_offset_of(
				    pack, layout, layer, expert, matrix,
				    &byte_count);
			    is_listed =
				    is_listed &&
				    entries[matrix] ==
					    (expert < first ?
						     offset :
						     card->state_offset +
							     (uint64_t)(expert -
									first) *
								     LINEAR_OFFSET_ALIGNMENT_BYTES);
		    }
		    uint64_t expert_end = 0;
		    is_listed =
			    is_listed &&
			    (expert < first ||
			     plan->host_expert_offsets[layer][expert - first] ==
				     expert_span_of(pack, layout, layer, expert,
						    &expert_end));
	    }
	    expect(is_listed,
		   "the table names each expert in host memory by its tag, the others where they lie");
	    uint64_t last_end = 0;
	    const uint64_t range_first =
		    first == 0 ? card->state_offset :
				 expert_span_of(pack, layout, layer, 0, &end);
	    (void)expert_span_of(pack, layout, layer, EXPERTS - 1u, &last_end);
	    const uint64_t range_end =
		    index < PAUSED_LAYERS ?
			    card->state_offset + card->slots_byte_count :
			    last_end;
	    expect(plan->experts[layer].byte_offset == range_first &&
			   plan->experts[layer].byte_count ==
				   range_end - range_first &&
			   (index < PAUSED_LAYERS ||
			    base->experts[layer].byte_count ==
				    plan->experts[layer].byte_count),
		   "a layer's experts' range runs from its first on the card, or the slots, to its last or the slots' end");
    }
}

static void test_plan(const struct pack_record *pack)
{
    static uint32_t card_of_unit[UNIT_COUNT];
    struct load_layout *layout = NULL, *base_layout = NULL;
    context_text = "the plan";
    struct model_plan_record *const plan = plan_of(pack, true, &layout);
    struct model_plan_record *const base = plan_of(pack, false, &base_layout);
    if (plan != NULL && base != NULL) {
        expect_plan(pack, layout, plan, base);
        placement_of(card_of_unit, true);
	const struct placement_record placement = { card_of_unit, UNIT_COUNT,
						    CARDS };
	for (uint32_t card = 0; card < CARDS; card++) {
            uint64_t state = 0;
	    expect_status(v41_state_byte_count(pack, &placement, &limits, card,
					       &state),
			  STATUS_OK, "a card's state is counted");
	    expect(state == plan->cards[card].arena_byte_count - plan->cards[card].state_offset,
                   "state_byte_count gives each card's state, its slots with it");
        }
        uint64_t host_state = 0;
	expect_status(v41_state_byte_count(pack, &placement, &limits,
					   PLACEMENT_IN_HOST_MEMORY,
					   &host_state),
		      STATUS_OK, "host memory is counted");
	expect(host_state == (uint64_t)(HOST_EXPERTS + EXPERTS) * plan->expert_byte_count,
               "host memory: the page-locked copies of the experts there");
    }
    v41_plan_destroy(plan);
    v41_plan_destroy(base);
    load_layout_destroy(layout);
    load_layout_destroy(base_layout);
}

/* ---- programs ---- */

static void cards_open(struct device_card **cards,
		       struct card_description_record *descriptions)
{
    test_host_memory_backend_reset(CARDS, HOST_CARD_BYTES);
    for (uint32_t card = 0; card < CARDS; card++) {
	    test_host_memory_backend_set_arithmetic_class(
		    card, card == PRO_CARD ? ARITHMETIC_CLASS_SM_120 :
					     ARITHMETIC_CLASS_SM_80);
	    expect_status(test_host_memory_backend.card_describe(
				  card, &descriptions[card]),
			  STATUS_OK, "a card is described");
	    expect_status(test_host_memory_backend.card_open(
				  card, HOST_CARD_BYTES, &cards[card]),
			  STATUS_OK, "a card is opened");
    }
}

/* The pause of paused layer index against the plan; its two segments, the first ending with the router. */
static void expect_pause(const struct model_plan_record *plan,
			 const struct execution_program *program,
			 uint32_t index, uint32_t host_experts_before)
{
    const uint32_t layer = paused_layers[index], card = plan->layer_card[layer];
    const struct expert_pause *const pause = &program->expert_pauses[index];
    const struct v41_card *const planned = &plan->cards[card];
    const bool is_shaped =
	    pause->card == card &&
	    pause->after_segment + 1u < program->segment_count &&
	    pause->jobs_offset == planned->array_offset[V41_DATA_ARRAY_JOBS] &&
	    pause->slots_offset == planned->state_offset &&
	    pause->slot_byte_count == plan->slot_byte_count &&
	    pause->slot_count == planned->slot_count &&
	    pause->expert_byte_count == plan->expert_byte_count &&
	    memcmp(pause->matrix_offsets, plan->expert_matrix_offsets,
		   sizeof(pause->matrix_offsets)) == 0 &&
	    pause->first_host_expert == host_experts_before &&
	    pause->host_expert_count == plan->host_expert_count[layer] &&
	    pause->jobs_per_row == MODEL_EXPERTS_PER_ROW;
    expect(is_shaped, "the pause names the layer's job table, the card's slots and the layer's host experts");
    expect(is_shaped && memcmp(&program->host_expert_offsets[pause->first_host_expert],
                               plan->host_expert_offsets[layer], pause->host_expert_count * sizeof(uint64_t)) == 0,
           "the program holds the layer's host experts' offsets");
    if (!is_shaped) {
        return;
    }
    const struct segment_record *const before =
	    &program->segments[pause->after_segment];
    const struct segment_record *const after = before + 1;
    expect(before->card == card && before->layer == layer &&
		   before->pause == SEGMENT_PAUSE_EXPERT_UPLOADS &&
		   after->card == card && after->layer == layer &&
		   after->pause == 0 &&
		   after->first_launch ==
			   before->first_launch + before->launch_count,
	   "the layer is two segments of its card, the first ending with the pause");
    const struct kernel_launch *const router =
	    &program->cards[card].launches[before->first_launch +
					   before->launch_count - 1u];
    struct test_host_memory_backend_kernel kernel;
    struct router_select_arguments record;
    memcpy(&record, router->argument, sizeof(record));
    expect(test_host_memory_backend_kernel_of(card, router->kernel, &kernel) &&
		   strcmp(kernel.function_name,
			  program->row_count == 1 ? KERNEL_FFN_FRONT_STEM "1" :
						    KERNEL_FFN_FRONT_STEM
				  "8") == 0 &&
		   router->argument_byte_count == SITE_FRONT_ARGUMENT_BYTES &&
		   record.row_count == program->row_count &&
		   record.jobs_offset == pause->jobs_offset &&
		   record.weights_offset == plan->experts[layer].byte_offset &&
		   record.expert_offsets_offset ==
			   planned->expert_table_offset[layer],
	   "the pause follows the fused FFN front's router, whose jobs name the layer's slots and expert table");
}

/* The step driver refuses copies of the programs that lay the n-gram rows of the first paused layer's card, or the
 * source of a hand-off from it, or the destination of one into the second's, over that card's slots. */
static void expect_slots_apart(const struct model_plan_record *plan,
			       struct execution_program *const *programs,
			       struct device_card **cards,
			       const struct engram_record *engram)
{
    enum { ENGRAM_ROWS, HAND_OFF_SOURCE, HAND_OFF_DESTINATION, CASES };
    static const char *const cases[CASES] = {"the n-gram rows over the slots", "a hand-off's source over the slots",
                                             "a hand-off's destination over the slots"};
    const uint32_t first = plan->layer_card[HOST_LAYER], second = plan->layer_card[ALL_HOST_LAYER];
    for (uint32_t index = 0; index < CASES; index++) {
	    static struct program_card_record areas[PROGRAMS][CARDS];
	    struct execution_program copies[PROGRAMS];
	    const struct execution_program *pointers[PROGRAMS];
	    struct hand_off_record *hand_offs[PROGRAMS] = { NULL };
	    bool is_changed[PROGRAMS] = { false };
	    context_text = cases[index];
	    for (uint32_t program = 0; program < PROGRAMS; program++) {
		    const uint32_t count = programs[program]->hand_off_count;
		    copies[program] = *programs[program];
		    memcpy(areas[program], programs[program]->cards,
			   sizeof(areas[program]));
		    hand_offs[program] =
			    calloc(count, sizeof(*hand_offs[program]));
		    expect(programs[program]->card_count == CARDS &&
				   hand_offs[program] != NULL,
			   "a copy of the program");
		    for (uint32_t hand_off = 0;
			 hand_offs[program] != NULL && hand_off < count;
			 hand_off++) {
			    struct hand_off_record *const copy =
				    &hand_offs[program][hand_off];
			    *copy = programs[program]->hand_offs[hand_off];
			    const bool is_it =
				    !is_changed[program] &&
				    (index == HAND_OFF_SOURCE ?
					     copy->from_card == first :
					     index == HAND_OFF_DESTINATION &&
						     copy->to_card == second);
			    copy->source_offset =
				    is_it && index == HAND_OFF_SOURCE ?
					    plan->cards[first].state_offset :
					    copy->source_offset;
			    copy->destination_offset =
				    is_it && index == HAND_OFF_DESTINATION ?
					    plan->cards[second].state_offset :
					    copy->destination_offset;
			    is_changed[program] = is_changed[program] || is_it;
		    }
		    if (index == ENGRAM_ROWS) {
			    areas[program][first].engram_rows_offset =
				    plan->cards[first].state_offset;
			    is_changed[program] =
				    areas[program][first].engram_tables != 0;
		    }
		    copies[program].cards = areas[program];
		    copies[program].hand_offs = hand_offs[program];
		    pointers[program] = &copies[program];
	    }
	    struct step_driver *driver = NULL;
	    expect(is_changed[0] && is_changed[1], "the programs are changed");
	    capture_begin();
	    expect_status(hand_offs[0] == NULL || hand_offs[1] == NULL ?
				  STATUS_INVALID_ARGUMENT :
				  step_driver_create(&test_host_memory_backend,
						     cards, pointers, PROGRAMS,
						     engram, WAIT_LIMIT,
						     &driver),
			  STATUS_INVALID_ARGUMENT,
			  "the step driver refuses the programs");
	    capture_end_expecting("step driver: refused");
	    expect(driver == NULL, "no driver is given");
	    for (uint32_t program = 0; program < PROGRAMS; program++) {
		    free(hand_offs[program]);
	    }
    }
}

static void test_programs(const struct pack_record *pack,
			  struct device_card **cards,
			  struct card_description_record *descriptions,
			  const struct engram_record *engram)
{
    static const uint32_t row_counts[] = {1u, ROWS};
    struct load_layout *layout = NULL, *base_layout = NULL;
    const struct model_operations *const model =
	    model_operations_of(V41_DATA_MODEL_NAME);
    struct execution_program *programs[COUNT_OF(row_counts)] = { NULL },
			     *base_programs[COUNT_OF(row_counts)] = { NULL };
    context_text = "the programs";
    struct model_plan_record *const plan = plan_of(pack, true, &layout);
    struct model_plan_record *const base = plan_of(pack, false, &base_layout);
    cards_open(cards, descriptions);
    for (uint32_t index = 0; model != NULL && plan != NULL && base != NULL && index < COUNT_OF(row_counts); index++) {
	    expect_status(
		    model->program_prepare(plan, &test_host_memory_backend,
					   cards, descriptions, PROGRAM_THIN,
					   row_counts[index], &programs[index]),
		    STATUS_OK, "the program is made");
	    expect_status(model->program_prepare(
				  base, &test_host_memory_backend, cards,
				  descriptions, PROGRAM_THIN, row_counts[index],
				  &base_programs[index]),
			  STATUS_OK,
			  "the program without host experts is made");
	    const struct execution_program *const program = programs[index],
						  *const other =
							  base_programs[index];
	    if (program == NULL || other == NULL) {
		    continue;
	    }
        expect(program->expert_pause_count == PAUSED_LAYERS && other->expert_pause_count == 0 &&
                   program->host_expert_count == HOST_EXPERTS + EXPERTS &&
                   program->segment_count == other->segment_count + PAUSED_LAYERS &&
                   program->hand_off_count == other->hand_off_count,
               "two pauses, their experts, a segment more for each, the same hand-offs");
        for (uint32_t pause = 0, before = 0; pause < PAUSED_LAYERS && pause < program->expert_pause_count; pause++) {
            expect_pause(plan, program, pause, before);
            before += program->expert_pauses[pause].host_expert_count;
        }
        for (uint32_t card = 0; card < CARDS; card++) {
            bool is_same = program->cards[card].launch_count == other->cards[card].launch_count;
            for (uint32_t launch = 0; is_same && launch < program->cards[card].launch_count; launch++) {
                is_same = program->cards[card].launches[launch].kernel == other->cards[card].launches[launch].kernel;
            }
            expect(is_same, "each card runs the kernels of the placement without host experts");
        }
    }
    struct step_driver *driver = NULL;
    expect_status(
	    programs[0] == NULL || programs[1] == NULL ?
		    STATUS_INVALID_ARGUMENT :
		    step_driver_create(
			    &test_host_memory_backend, cards,
			    (const struct execution_program *const *)programs,
			    COUNT_OF(row_counts), engram, WAIT_LIMIT, &driver),
	    STATUS_OK, "the step driver takes the programs");
    step_driver_destroy(driver);
    if (programs[0] != NULL && programs[1] != NULL && plan != NULL) {
        expect_slots_apart(plan, programs, cards, engram);
    }
    for (uint32_t index = 0; model != NULL && index < COUNT_OF(row_counts); index++) {
        model->program_destroy(programs[index]);
        model->program_destroy(base_programs[index]);
    }
    for (uint32_t card = 0; card < CARDS; card++) {
        test_host_memory_backend.card_close(cards[card], 0);
        cards[card] = NULL;
    }
    v41_plan_destroy(plan);
    v41_plan_destroy(base);
    load_layout_destroy(layout);
    load_layout_destroy(base_layout);
}

/* ---- refusals ---- */

static void test_refusals(void)
{
    static uint32_t card_of_unit[UNIT_COUNT];
    static const struct {
        const char *name; uint32_t unit; enum layout_change layout; const char *what;
    } cases[] = {
        {"layers.5.ffn.experts.383.w2.weight", TEST_MODEL_V41_EXPERT_UNIT(HOST_LAYER, NEIGHBOUR_EXPERT), LAYOUT_SAME,
         "an expert in host memory with a matrix apart from the others"},
        {"layers.5.ffn.experts.383.w2.weight", TEST_MODEL_V41_EXPERT_UNIT(HOST_LAYER, 0), LAYOUT_SAME,
         "an expert in host memory with a matrix on the card"},
        {"layers.5.ffn.experts.383.w1.weight", TEST_MODEL_V41_EXPERT_UNIT(HOST_LAYER, NEIGHBOUR_EXPERT),
         LAYOUT_REORDERED, "an expert in host memory with its matrices together in another order"},
        {NULL, 0, LAYOUT_BEYOND, "an expert in host memory beyond the host arena"}};
    placement_of(card_of_unit, true);
    for (uint32_t index = 0; index < COUNT_OF(cases); index++) {
	    struct pack_record *pack = NULL;
	    struct load_layout *layout = NULL;
	    struct model_plan_record *plan = NULL;
	    const struct test_model_v41_change change = {
		    cases[index].name != NULL ? TEST_MODEL_V41_CHANGE_UNIT :
						TEST_MODEL_V41_CHANGE_NONE,
		    cases[index].name, cases[index].unit
	    };
	    context_text = cases[index].what;
	    expect_status(test_model_v41_pack_open(change, &pack), STATUS_OK,
			  "the changed pack opens");
	    result_status status =
		    pack == NULL ?
			    STATUS_INVALID_ARGUMENT :
			    load_layout_create(pack, card_of_unit, UNIT_COUNT,
					       CARDS, &layout);
	    expect_status(status, STATUS_OK,
			  "the loader lays the placement out");
	    /* the down and up of the last expert lie in its own extent, down first: from the gate on, aligned */
	    const struct pack_tensor *const up =
		    layout == NULL ? NULL :
				     matrix_of(pack, HOST_LAYER, EXPERTS - 1u,
					       V41_DATA_UP);
	    if (up != NULL && cases[index].layout != LAYOUT_SAME) {
		    uint64_t gate_bytes = 0;
		    const uint64_t gate = matrix_offset_of(
			    pack, layout, HOST_LAYER, EXPERTS - 1u,
			    V41_DATA_GATE, &gate_bytes);
		    const uint64_t up_end =
			    up->byte_offset_in_extent + up->byte_count;
		    const uint64_t aligned_end =
			    (up_end + WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES -
			     1u) /
			    WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES *
			    WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES;
		    expect(gate >= aligned_end,
			   "the gate lies behind where the moved extent begins");
		    layout->byte_offset_of_extent[up->extent_index] =
			    cases[index].layout == LAYOUT_BEYOND ?
				    layout->host_byte_count :
				    gate - aligned_end;
	    }
        capture_begin();
	status = layout == NULL ? status :
				  v41_plan_create(pack, layout, &limits, &plan);
	capture_end_expecting("expert 383 of layer 5 is neither");
	expect_status(status, STATUS_UNSUPPORTED, cases[index].what);
	expect(plan == NULL, "no plan is given");
	v41_plan_destroy(plan);
	load_layout_destroy(layout);
	pack_close(pack);
    }
}

int main(void)
{
	struct device_card *cards[CARDS] = { NULL };
	struct card_description_record descriptions[CARDS];
	struct pack_record *pack = NULL;
	struct engram_record *engram = NULL;
	unsigned char *engram_pack = NULL;
	(void)setvbuf(stdout, NULL, _IOLBF,
		      0); /* a failure's line survives a sanitizer's stop */
	capture_descriptor = mkstemp(capture_path);
	error_stream_descriptor = dup(STDERR_FILENO);
	if (capture_descriptor < 0 || error_stream_descriptor < 0 ||
	    unlink(capture_path) != 0) {
		(void)printf(
			"FAILED: the temporary file for the error stream\n");
		return EXIT_FAILURE;
	}
    context_text = "the model's pack";
    test_refusals();
    expect_status(test_model_v41_pack_open(
			  (struct test_model_v41_change){
				  TEST_MODEL_V41_CHANGE_NONE, NULL, 0 },
			  &pack),
		  STATUS_OK, "the model's pack opens from its tables");
    if (pack != NULL) {
        test_plan(pack);
    }
    /* the n-gram tables the driver checks the programs' cards against: the pack reads nothing once open */
    expect_status(test_engram_pack_open(&engram_pack, &engram), STATUS_OK,
		  "the n-gram tables open");
    if (pack != NULL && engram != NULL) {
        test_programs(pack, cards, descriptions, engram);
    }
    engram_close(engram);
    free(engram_pack);
    pack_close(pack);
    test_memory_storage_reset();
    (void)close(capture_descriptor);
    (void)close(error_stream_descriptor);
    if (failures != 0) {
        (void)printf("model_v41_host_experts_test FAILED: %d failures\n", failures);
        return EXIT_FAILURE;
    }
    (void)printf("model_v41_host_experts_test passed\n");
    return EXIT_SUCCESS;
}
