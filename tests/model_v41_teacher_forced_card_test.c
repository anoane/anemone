/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * model_v41_teacher_forced_card_test: the teacher-forced harness (teacher_forced.h) on the real cards and pack, for the
 * layer gate of the first milestone (first_token_plan.md 3.3, 4.2: unit U14).
 *
 *     model_v41_teacher_forced_card_test [--layers=0-39] [--positions=N] [--free] [--logits] [--order=reference]
 *         [--numerics=exact|relaxed|fast] [--pro_emulation=0|1] [--layout=TEXT (all layers only)]
 *         [--profile=speed] [--card_order=0,1,3,2] [--card_memory_kept_free_mib=1024] [--break=pre] [--label=TEXT]
 *         <capture folder> <output folder> <pack copy> [<pack copy> ...]
 *
 *   - The capture folder is one of tools/oracle/stepwise_capture.py --mode stepwise; the output folder, which must not
 *     exist, receives the engine's records for tools/oracle/compare_engine_capture.py --gate layer (injected, the
 *     default) or, with --free --logits, --gate teacher_forced (the tokens forced, the engine's own streams).
 *   - --layers: the layers run and written, ranges a-b and single layers separated by commas (default all); the
 *     source and the index layer of each run with it, as their pools, keys and selections feed it. Every other layer
 *     is loaded with what fits of its experts and never runs (subset_placement.h). So four cards run what fits until
 *     the fifth card is in.
 *   - Placement: the cards in --card_order (backend indices; default all, as enumerated), each given its free memory
 *     less the backend's reserve and --card_memory_kept_free_mib (default 1024, which also holds the model's state),
 *     filled as subset_placement.h says.
 *   - --pro_emulation defaults to 1 for legacy oracle runs; explicit 0 selects the serving native PRO arithmetic.
 *     --layout pins a complete-layer PP baseline before loading, including the PRO tail. Conditions retain the
 *     actual card_of_layer map, numerical mode and emulation flag; no requested layout is inferred from card_order.
 *   - --label: up to 63 letters, digits and ._:- that the conditions of the manifest hold as "label" (the batch
 *     gives its snapshot and a digest of the engine's sources), null without it.
 *   - The pack loads (loader.h), the n-gram tables are attached, the state is set and the program prepared for steps
 *     of one row in the --order (default fixed) and --profile (default precision) asked, for the context of the first
 *     milestone; the harness then runs the first --positions (default all) of the capture.
 *   - --break=pre: layer TEST_BREAK_LAYER (the first of the third card in first_token_plan.md 1.3) is given its
 *     carried pre as zeros at every step: the layer gate must fail at that layer alone.
 *   - Without arguments, as `make card_tests` runs every card test, a pattern goes up to every card and back, and it
 *     says that nothing was teacher-forced.
 * A failure is one line on the standard error stream; the exit status is 0 only when the folder was written complete,
 * and TEST_EXIT_DOES_NOT_FIT when the layers asked did not fit the cards (placed, or opened), before any load.
 */
#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "anemone/engram.h"
#include "anemone/hooks.h"
#include "anemone/loader.h"
#include "anemone/model_seam.h"
#include "anemone/official_reference.h"
#include "anemone/pack.h"
#include "anemone/placement.h"
#include "anemone/status.h"
#include "anemone/storage_seam.h"
#include "anemone/startup_policy.h"
#include "capture_reader.h"
#include "model_v41.h"
#include "subset_placement.h"
#include "teacher_forced.h"

#define TEST_TIME_LIMIT_SECONDS 14400                  /* four hours: the 2,048 positions of 40 layers take minutes */
#define WAIT_LIMIT_NANOSECONDS 120000000000ull         /* two minutes for any one wait */
#define CARD_CLOSE_WAIT_LIMIT_NANOSECONDS 10000000000ull
#define TEST_BREAK_LAYER 16u
#define DEFAULT_KEPT_FREE_MIB 1024u
#define MOST_KEPT_FREE_MIB 1048576u                    /* a TiB: above any card */
#define BYTES_PER_MIB 1048576ull
#define BYTES_PER_GIB 1073741824.0
#define NANOSECONDS_PER_SECOND 1e9
#define DECIMAL_BASE 10
#define LABEL_CAPACITY 64u
#define LABEL_CHARACTERS "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-"
#define CONDITIONS_CAPACITY 2048u
#define TEST_EXIT_DOES_NOT_FIT 3
#define ROUNDTRIP_BYTES 1048576ull
#define ROUNDTRIP_STAGING_BYTES 2097152ull             /* what goes up, then what comes back */
#define FOLDERS 2                                      /* the capture folder and the output folder */
#define ROUNDTRIP_SEED 0x9E3779B9u
#define ROUNDTRIP_SHIFT 11u
#define EVERY_LAYER ((1ull << MODEL_LAYERS) - 1u)
#define BIT(index) (1ull << (index))

struct options {
    const char *layout;
    const char *capture_folder;
    const char *output_folder;
    const char *const *copy_paths;
    const char *label;                                 /* or NULL */
    uint64_t layers;                                   /* asked; 0 until given */
    uint32_t card_order[BACKEND_MAXIMUM_CARDS];
    uint32_t card_count;                               /* 0: every card, as enumerated */
    uint32_t copy_count;
    uint32_t positions;                                /* 0: every position of the capture */
    uint32_t kept_free_mib;
    uint32_t official_hc_rows;
    uint32_t official_reference;
    uint32_t pro_emulation;                            /* default 1 preserves the legacy oracle harness */
    enum numerics_code numerics;
    enum linear_order_code order;
    enum linear_profile_code profile;
    bool free_running;
    bool logits;
    bool broken;
};

/* What a run holds, released in reverse by run_release. */
struct load {
	struct storage_reader *reader;
	struct pack_record *pack;
	struct engram_record *engram;
	struct load_layout *layout;
	struct test_subset_placement
		placement; /* with the layout's copy the plan reads */
	struct model_plan_record *plan;
	struct execution_program *program;
	const struct model_operations *model;
	struct device_card *cards[BACKEND_MAXIMUM_CARDS];
	struct card_description_record descriptions[BACKEND_MAXIMUM_CARDS];
	struct load_result result;
	uint64_t budget[BACKEND_MAXIMUM_CARDS];
	uint64_t room[BACKEND_MAXIMUM_CARDS]; /* free memory less the backend's reserve */
};

/* Bytes as GiB, for a line of the run. */
static double gib_of(uint64_t byte_count)
{
    const double gib = (double)byte_count / BYTES_PER_GIB;
    return gib;
}

/* "a-b,c" into bits, every number a layer of the model, each range ascending. */
static bool layers_parse(const char *text, uint64_t *layers_out)
{
    uint64_t layers = 0;
    char *end = NULL;
    for (const char *next = text; *next >= '0' && *next <= '9'; next = end + 1) {
        errno = 0;
        const unsigned long long first = strtoull(next, &end, DECIMAL_BASE);
        unsigned long long last = first;
        if (*end == '-' && end[1] >= '0' && end[1] <= '9') {
            last = strtoull(end + 1, &end, DECIMAL_BASE);
        }
	if (errno != 0 || first > last || last >= MODEL_LAYERS) {
		return false;
	}
	for (unsigned long long layer = first; layer <= last; layer++) {
            layers |= BIT(layer);
        }
        if (*end == '\0') {
            *layers_out = layers;
            return true;
        }
        if (*end != ',') {
            return false;
        }
    }
    return false;
}

/* A number of the option name=, from least to most, given once (*given false before). */
static bool number_take(const char *argument, const char *name, uint32_t least, uint32_t most, uint32_t *value_out,
                        bool *given)
{
    const size_t length = strlen(name);
    char *end = NULL;
    if (strncmp(argument, name, length) != 0 || *given || argument[length] < '0' || argument[length] > '9') {
        return false;
    }
    errno = 0;
    const unsigned long long value = strtoull(argument + length, &end, DECIMAL_BASE);
    *given = errno == 0 && *end == '\0' && value >= least && value <= most;
    *value_out = *given ? (uint32_t)value : *value_out;
    return *given;
}

/* "1,0,2": backend cards, none twice. */
static bool card_order_parse(const char *text, struct options *options)
{
    char *end = NULL;
    for (const char *next = text; options->card_count < BACKEND_MAXIMUM_CARDS &&
				  *next >= '0' && *next <= '9';
	 next = end + 1) {
	    errno = 0;
	    const unsigned long long card = strtoull(next, &end, DECIMAL_BASE);
	    for (uint32_t earlier = 0; earlier < options->card_count;
		 earlier++) {
		    end = options->card_order[earlier] == card ? NULL : end;
	    }
	    if (end == NULL || errno != 0 || card >= BACKEND_MAXIMUM_CARDS) {
		    return false;
	    }
	    options->card_order[options->card_count++] = (uint32_t)card;
	    if (*end == '\0') {
		    return true;
	    }
	    if (*end != ',') {
		    return false;
	    }
    }
    return false;
}

static bool options_parse(int argument_count, char **arguments, struct options *options)
{
    bool positions_given = false, kept_given = false, order_given = false, profile_given = false, break_given = false;
    bool numerics_given = false, pro_given = false, hc_given = false,
	 official_given = false;
    int index = 1;
    *options = (struct options){ .kept_free_mib = DEFAULT_KEPT_FREE_MIB,
				 .pro_emulation = 1,
				 .order = LINEAR_ORDER_FIXED,
				 .profile = LINEAR_PROFILE_PRECISION };
    for (; index < argument_count && strncmp(arguments[index], "--", strlen("--")) == 0; index++) {
        const char *const argument = arguments[index];
        const bool is_flag_again = (strcmp(argument, "--free") == 0 && options->free_running) ||
                                   (strcmp(argument, "--logits") == 0 && options->logits);
        bool taken = !is_flag_again && (strcmp(argument, "--free") == 0 || strcmp(argument, "--logits") == 0);
        options->free_running = options->free_running || (taken && strcmp(argument, "--free") == 0);
        options->logits = options->logits || (taken && strcmp(argument, "--logits") == 0);
        taken = taken || number_take(argument, "--positions=", 1, INT32_MAX, &options->positions, &positions_given) ||
                number_take(argument, "--pro_emulation=", 0, 1, &options->pro_emulation, &pro_given) ||
                number_take(argument, "--card_memory_kept_free_mib=", 0, MOST_KEPT_FREE_MIB, &options->kept_free_mib,
                            &kept_given);
	taken = taken || number_take(argument, "--official_hc_rows=", 0,
				     LINEAR_MAXIMUM_ROWS,
				     &options->official_hc_rows, &hc_given);
	taken = taken ||
		number_take(argument, "--official_reference=", 0, 1,
			    &options->official_reference, &official_given);
	if (!taken && options->layout == NULL &&
	    strncmp(argument, "--layout=", strlen("--layout=")) == 0) {
		options->layout = argument + strlen("--layout=");
		taken = options->layout[0] != '\0';
	} else if (!taken && !numerics_given &&
		   strncmp(argument, "--numerics=", strlen("--numerics=")) ==
			   0) {
		const result_status parsed = numerics_from_text(
			argument + strlen("--numerics="), &options->numerics);
		taken = numerics_given = parsed == STATUS_OK;
	} else if (!taken &&
		   strncmp(argument, "--layers=", strlen("--layers=")) == 0 &&
		   options->layers == 0) {
		taken = layers_parse(argument + strlen("--layers="),
				     &options->layers);
	} else if (!taken &&
		   strncmp(argument,
			   "--card_order=", strlen("--card_order=")) == 0 &&
		   options->card_count == 0) {
		taken = card_order_parse(argument + strlen("--card_order="),
					 options);
	} else if (!taken && !order_given &&
		   (strcmp(argument, "--order=reference") == 0 ||
		    strcmp(argument, "--order=fixed") == 0)) {
		options->order = strcmp(argument, "--order=fixed") == 0 ?
					 LINEAR_ORDER_FIXED :
					 LINEAR_ORDER_REFERENCE;
		taken = order_given = true;
	} else if (!taken && !profile_given &&
		   (strcmp(argument, "--profile=precision") == 0 ||
		    strcmp(argument, "--profile=speed") == 0)) {
		options->profile = strcmp(argument, "--profile=speed") == 0 ?
					   LINEAR_PROFILE_SPEED :
					   LINEAR_PROFILE_PRECISION;
		taken = profile_given = true;
	} else if (!taken && !break_given &&
		   strcmp(argument, "--break=pre") == 0) {
		taken = break_given = options->broken = true;
	} else if (!taken && options->label == NULL &&
		   strncmp(argument, "--label=", strlen("--label=")) == 0) {
		const char *const label = argument + strlen("--label=");
		const size_t length = strnlen(label, LABEL_CAPACITY);
		taken = length != 0 && length < LABEL_CAPACITY &&
			strspn(label, LABEL_CHARACTERS) == length;
		options->label = taken ? label : NULL;
	}
	if (!taken) {
            return false;
        }
    }
    const int copies = argument_count - index - FOLDERS;
    if (copies < 1 || copies > STORAGE_MAXIMUM_COPIES ||
	(options->free_running && options->broken)) {
	    return false;
    }
    options->capture_folder = arguments[index];
    options->output_folder = arguments[index + 1];
    options->copy_paths = (const char *const *)&arguments[index + FOLDERS];
    options->copy_count = (uint32_t)copies;
    options->layers = options->layers == 0 ? EVERY_LAYER : options->layers;
    return true;
}

/* The cards: described in the order asked, each budget its free memory less the reserve and what is kept free. */
MUST_CHECK static result_status cards_describe(struct load *load,
					       struct options *options)
{
	const struct backend_operations *const backend = &backend_cuda_value;
	uint32_t found = 0;
	result_status status = backend->card_count(&found);
	status = status == STATUS_OK &&
				 (found == 0 || found > BACKEND_MAXIMUM_CARDS) ?
			 STATUS_UNSUPPORTED :
			 status;
	for (uint32_t card = 0;
	     status == STATUS_OK && options->card_count == 0 && card < found;
	     card++) {
		options->card_order[card] = card;
	}
	options->card_count = options->card_count == 0 && status == STATUS_OK ?
				      found :
				      options->card_count;
	for (uint32_t card = 0;
	     status == STATUS_OK && card < options->card_count; card++) {
		status = options->card_order[card] >= found ?
				 STATUS_INVALID_ARGUMENT :
				 backend->card_describe(
					 options->card_order[card],
					 &load->descriptions[card]);
		if (status == STATUS_OK && options->pro_emulation == 0 &&
		    load->descriptions[card].arithmetic_class ==
			    ARITHMETIC_CLASS_SM_120)
			load->descriptions[card].arithmetic_class =
				ARITHMETIC_CLASS_SM_120_NATIVE;
		const struct card_description_record *const description =
			&load->descriptions[card];
		const uint64_t kept =
			(uint64_t)options->kept_free_mib * BYTES_PER_MIB;
		load->room[card] =
			description->memory_free_bytes >
					description->memory_reserved_bytes ?
				description->memory_free_bytes -
					description->memory_reserved_bytes :
				0;
		load->budget[card] =
			load->room[card] > kept ? load->room[card] - kept : 0;
	}
	if (status != STATUS_OK) {
		LOG_ERROR(
			"the cards could not be counted or described (%" PRIu32
			" found): %s",
			found, status_text(status));
	}
    return status;
}

/* Every card the plan gives an arena opens with exactly it, if the card has the room. */
MUST_CHECK static result_status cards_open(struct load *load,
					   const struct options *options)
{
	result_status status = STATUS_OK;
	for (uint32_t card = 0;
	     status == STATUS_OK && card < options->card_count; card++) {
		uint64_t arena = 0;
		status =
			load->model->arena_byte_count(load->plan, card, &arena);
		if (status == STATUS_OK && arena > load->room[card]) {
			LOG_ERROR(
				"card %" PRIu32
				": weights and state take %.3f GiB, it has %.3f",
				options->card_order[card], gib_of(arena),
				gib_of(load->room[card]));
			status = STATUS_OUT_OF_MEMORY;
		}
		status = status != STATUS_OK || arena == 0 ?
				 status :
				 backend_cuda_value.card_open(
					 options->card_order[card], arena,
					 &load->cards[card]);
		if (status == STATUS_OK) {
			(void)printf(
				"card %" PRIu32
				": arena %.3f GiB, weights %.3f GiB\n",
				options->card_order[card], gib_of(arena),
				gib_of(load->layout->card_byte_count[card]));
		}
	}
    return status;
}

/* The n-gram tables, from host memory where the loader put them; table t is the t-th n-gram layer. */
MUST_CHECK static result_status
engram_attach(struct load *load, struct test_teacher_forced_run *run)
{
	const struct engram_description *const engram =
		engram_description_of(load->engram);
	const struct pack_unit_record_2 *const units =
		pack_units_of(load->pack);
	result_status status =
		engram->table_count == (uint32_t)__builtin_popcountll(
					       V41_DATA_ENGRAM_LAYERS) &&
				engram->rows_per_token ==
					V41_DATA_ENGRAM_ROWS_PER_TOKEN &&
				load->result.host_arena != NULL ?
			STATUS_OK :
			STATUS_CORRUPT_PACK;
	for (uint32_t table = 0, layer = 0;
	     status == STATUS_OK && table < engram->table_count;
	     table++, layer++) {
		const uint32_t unit = engram->table_unit_index[table];
		const uint32_t extent =
			unit < pack_header_of(load->pack)->unit_count ?
				units[unit].first_extent_index :
				UINT32_MAX;
		while ((V41_DATA_ENGRAM_LAYERS & BIT(layer)) == 0) {
			layer++;
		}
		run->engram_layers[table] = layer;
		status = extent == UINT32_MAX ||
					 load->layout->card_of_extent[extent] !=
						 PLACEMENT_IN_HOST_MEMORY ?
				 STATUS_CORRUPT_PACK :
				 engram_attach_entry(
					 load->engram, table,
					 (const unsigned char *)load->result
							 .host_arena +
						 load->layout
							 ->byte_offset_of_extent
								 [extent]);
	}
	if (status != STATUS_OK) {
		LOG_ERROR(
			"the n-gram tables are not the model's, or not in host memory: %s",
			status_text(status));
	}
    return status;
}

/* The run's conditions for the manifest: the settings, the label, and which card runs each layer. */
static bool conditions_write(char *text, const struct options *options, const struct load *load, uint64_t layers)
{
	int length = snprintf(
		text, CONDITIONS_CAPACITY,
		"{\"capture\": \"teacher_forced\", \"mode\": \"%s\", "
		"\"linear_order\": \"%s\", \"linear_profile\": \"%s\", \"numerics\": \"%s\", "
		"\"pro_emulation\": %u, \"explicit_layout\": %u, \"context_token_limit\": %u, "
		"\"break\": %s, \"label\": %s%s%s, \"layers_run\": %" PRIu64
		", \"card_of_layer\": [",
		options->free_running ? "free" : "injected",
		options->order == LINEAR_ORDER_REFERENCE ? "reference" :
							   "fixed",
		options->profile == LINEAR_PROFILE_SPEED ? "speed" :
							   "precision",
		options->numerics == NUMERICS_EXACT   ? "exact" :
		options->numerics == NUMERICS_RELAXED ? "relaxed" :
							"fast",
		options->pro_emulation, options->layout != NULL ? 1u : 0u,
		(options->official_reference != 0 ?
			 OFFICIAL_REFERENCE_CONTEXT_LIMIT :
			 MODEL_CONTEXT_WITHOUT_CANDIDATES),
		options->broken ? "\"pre\"" : "null",
		options->label == NULL ? "" : "\"",
		options->label == NULL ? "null" : options->label,
		options->label == NULL ? "" : "\"", layers);
	for (uint32_t layer = 0;
	     length > 0 && length < (int)CONDITIONS_CAPACITY &&
	     layer <= MODEL_LAYERS;
	     layer++) {
		const int more =
			layer == MODEL_LAYERS ?
				snprintf(text + length,
					 CONDITIONS_CAPACITY - (size_t)length,
					 "]}") :
				snprintf(
					text + length,
					CONDITIONS_CAPACITY - (size_t)length,
					"%s%" PRIu32, layer == 0 ? "" : ", ",
					options->card_order
						[load->plan->layer_card[layer]]);
		length = more < 0 ? -1 : length + more;
	}
    return length > 0 && length < (int)CONDITIONS_CAPACITY;
}

static void load_release(struct load *load)
{
    if (load->model != NULL && load->program != NULL) {
        load->model->program_destroy(load->program);
    }
    for (uint32_t card = 0; card < BACKEND_MAXIMUM_CARDS; card++) {
	    backend_cuda_value.card_close(load->cards[card],
					  CARD_CLOSE_WAIT_LIMIT_NANOSECONDS);
    }
    if (load->layout != NULL) {                         /* behind the cards: an upload may have read it until then */
	    load_host_arena_release(load->result.host_arena,
				    load->layout->host_byte_count);
    }
    if (load->model != NULL && load->plan != NULL) {
        load->model->plan_destroy(load->plan);
    }
    test_subset_placement_release(&load->placement);
    load_layout_destroy(load->layout);
    engram_close(load->engram);
    pack_close(load->pack);
    storage_direct.reader_close(load->reader);
}

/* Up to the program: the pack, the placement, the load, the state; prints a line a step. does_not_fit_out: the
 * layers did not fit the cards, placed or opened. */
MUST_CHECK static result_status
load_run(struct load *load, struct options *options, uint64_t layers,
	 struct test_teacher_forced_run *run, bool *does_not_fit_out)
{
	const struct model_limits limits = {
		.context_token_limit =
			options->official_reference != 0 ?
				OFFICIAL_REFERENCE_CONTEXT_LIMIT :
				MODEL_CONTEXT_WITHOUT_CANDIDATES,
		.row_capacity = 1,
		.session_capacity = 1,
		.state_width = STATE_WIDTH_16,
		.linear_profile = options->profile,
		.linear_order = options->order,
		.cmp_order = options->numerics == NUMERICS_EXACT ?
				     CMP_ORDER_REFERENCE :
				     CMP_ORDER_RELAXED,
		.official_hc_rows = options->official_hc_rows,
		.official_reference = options->official_reference
	};
	result_status status = storage_direct.reader_open(
		options->copy_paths, options->copy_count, &load->reader);
	status = status != STATUS_OK ?
			 status :
			 pack_open_entry(&storage_direct, load->reader,
					 &load->pack);
	load->model = status == STATUS_OK ?
			      model_operations_of(
				      pack_header_of(load->pack)->model_name) :
			      NULL;
	status = status != STATUS_OK || load->model != NULL ?
			 status :
			 STATUS_UNSUPPORTED;
	status = status != STATUS_OK ?
			 status :
			 engram_open(&storage_direct, load->reader, load->pack,
				     &load->engram);
	status = status != STATUS_OK ? status : cards_describe(load, options);
	status = status != STATUS_OK ?
			 status :
			 test_subset_place(load->pack, load->budget,
					   options->card_count, layers,
					   &load->placement);
	if (status == STATUS_OK && options->layout != NULL) {
		status = layers != EVERY_LAYER ?
				 STATUS_UNSUPPORTED :
				 placement_from_layout_text(
					 load->pack, options->layout,
					 load->budget, options->card_count,
					 pack_header_of(load->pack)->unit_count,
					 load->placement.card_of_unit);
	}
	*does_not_fit_out = status == STATUS_OUT_OF_MEMORY &&
			    load->placement.card_of_unit != NULL &&
			    load->placement.layer_of_unit != NULL;
	status = status != STATUS_OK ?
			 status :
			 load_layout_create(
				 load->pack, load->placement.card_of_unit,
				 pack_header_of(load->pack)->unit_count,
				 options->card_count, &load->layout);
	status = status != STATUS_OK ?
			 status :
			 test_subset_plan_layout(load->pack, load->layout,
						 &load->placement);
	status = status != STATUS_OK ?
			 status :
			 load->model->plan_create(load->pack,
						  &load->placement.plan_layout,
						  &limits, &load->plan);
	if (status == STATUS_OK) {
		status = cards_open(load, options);
		*does_not_fit_out = status == STATUS_OUT_OF_MEMORY;
	}
	(void)printf("model_v41_teacher_forced_card_test: placed on %" PRIu32
		     " cards, %.2f GiB in host memory: %s\n",
		     options->card_count,
		     load->layout == NULL ?
			     0.0 :
			     gib_of(load->layout->host_byte_count),
		     status_text(status));
	const struct load_request request = {
		.storage = &storage_direct,
		.reader = load->reader,
		.backend = &backend_cuda_value,
		.pack = load->pack,
		.layout = load->layout,
		.cards = load->cards,
		.card_descriptions = load->descriptions,
		.reader_threads_per_copy =
			LOADER_DEFAULT_READER_THREADS_PER_COPY,
		.stall_limit_nanoseconds =
			LOADER_DEFAULT_STALL_LIMIT_NANOSECONDS
	};
	status = status != STATUS_OK ? status :
				       load_run_entry(&request, &load->result);
	status = status != STATUS_OK ? status : engram_attach(load, run);
	status = status != STATUS_OK ?
			 status :
			 load->model->arena_initialise(load->plan,
						       &backend_cuda_value,
						       load->cards,
						       WAIT_LIMIT_NANOSECONDS);
	status = status != STATUS_OK ? status :
				       load->model->program_prepare(
					       load->plan, &backend_cuda_value,
					       load->cards, load->descriptions,
					       PROGRAM_THIN, 1, &load->program);
	(void)printf(
		"model_v41_teacher_forced_card_test: loaded in %.1f s, state set, program of %" PRIu32
		" segments: "
		"%s\n",
		(double)load->result.transfer_nanoseconds /
			NANOSECONDS_PER_SECOND,
		load->program == NULL ? 0 : load->program->segment_count,
		status_text(status));
	return status;
}

/* Without arguments: a pattern up to every card and back, through the harness's backend. */
static int cards_touch(void)
{
	const struct backend_operations *const backend = &backend_cuda_value;
	uint32_t card_count = 0;
	void *block = NULL;
	int failures = 0;
	if (backend->card_count(&card_count) != STATUS_OK ||
	    card_count > BACKEND_MAXIMUM_CARDS ||
	    backend->staging_allocate(ROUNDTRIP_STAGING_BYTES, &block) !=
		    STATUS_OK) {
		(void)fprintf(
			stderr,
			"FAILED: the cards could not be counted, or no staging memory\n");
		return EXIT_FAILURE;
	}
    unsigned char *const sent = block, *const received = sent + ROUNDTRIP_BYTES;
    for (uint32_t card_index = 0; card_index < card_count; card_index++) {
	    struct device_card *card = NULL;
	    uint64_t ticket = 0;
	    for (uint64_t index = 0; index < ROUNDTRIP_BYTES; index++) {
		    sent[index] = (unsigned char)((index * ROUNDTRIP_SEED +
						   card_index) >>
						  ROUNDTRIP_SHIFT);
	    }
        memset(received, 0, ROUNDTRIP_BYTES);
	const bool held =
		backend->card_open(card_index, ROUNDTRIP_BYTES, &card) ==
			STATUS_OK &&
		backend->upload_begin(card, 0, sent, ROUNDTRIP_BYTES,
				      &ticket) == STATUS_OK &&
		backend->download_begin(card, 0, received, ROUNDTRIP_BYTES,
					&ticket) == STATUS_OK &&
		backend->upload_wait(card, ticket, WAIT_LIMIT_NANOSECONDS) ==
			STATUS_OK &&
		memcmp(sent, received, ROUNDTRIP_BYTES) == 0;
	backend->card_close(card, CARD_CLOSE_WAIT_LIMIT_NANOSECONDS);
        if (!held) {
            (void)fprintf(stderr, "FAILED: card %" PRIu32 " did not give back what went up\n", card_index);
            failures++;
        }
    }
    backend->staging_free(block);
    (void)printf("model_v41_teacher_forced_card_test: no capture and pack, nothing was teacher-forced; a pattern went "
                 "to %" PRIu32 " cards and back%s\n", card_count, failures == 0 ? "" : " NOT whole");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argument_count, char **arguments)
{
    struct options options;
    struct load load = {.reader = NULL};
    struct test_capture *capture = NULL;
    char conditions[CONDITIONS_CAPACITY];
    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    (void)setvbuf(stdout, NULL, _IOLBF, 0);
    if (argument_count == 1) {
        return cards_touch();
    }
    if (!options_parse(argument_count, arguments, &options)) {
	    (void)fprintf(
		    stderr,
		    "usage: model_v41_teacher_forced_card_test [--layers=0-39] [--positions=N] [--free] "
		    "[--logits] [--order=fixed|reference] [--profile=precision|speed] [--card_order=0,1,2] "
		    "[--numerics=exact|relaxed|fast] [--pro_emulation=0|1] [--layout=TEXT] "
		    "[--card_memory_kept_free_mib=N] [--break=pre] [--label=TEXT] <capture folder> "
		    "<output folder> <pack copy> [... up to %d copies]\n",
		    STORAGE_MAXIMUM_COPIES);
	    return EXIT_FAILURE;
    }
    const uint64_t layers = options.free_running ? EVERY_LAYER : test_subset_layers_with_sources(options.layers);
    bool does_not_fit = false;
    struct test_teacher_forced_run run = {
	    .backend = &backend_cuda_value,
	    .cards = load.cards,
	    .folder = options.output_folder,
	    .card_descriptions = load.descriptions,
	    .conditions = conditions,
	    .run_layers = layers,
	    .wait_limit_nanoseconds = WAIT_LIMIT_NANOSECONDS,
	    .layer_count = MODEL_LAYERS,
	    .positions = options.positions,
	    .break_layer = options.broken ? TEST_BREAK_LAYER :
					    TEST_TEACHER_FORCED_NO_BREAK,
	    .mode = options.free_running ? TEST_TEACHER_FORCED_FREE :
					   TEST_TEACHER_FORCED_INJECTED,
	    .logits = options.logits
    };
    result_status status = test_capture_open(options.capture_folder, &capture);
    const struct test_capture_tensor *const tokens = capture == NULL ? NULL : test_capture_find(capture,
                                                                                                "input.token_ids");
    const uint64_t captured = tokens == NULL || tokens->dimension_count == 0 ? 0
                            : tokens->dimensions[tokens->dimension_count - 1u];
    run.positions = options.positions == 0 && captured <= INT32_MAX ? (uint32_t)captured : run.positions;
    if (status == STATUS_OK &&
	(run.positions == 0 || run.positions > captured ||
	 run.positions > MODEL_CONTEXT_WITHOUT_CANDIDATES ||
	 (options.broken && (layers & BIT(TEST_BREAK_LAYER)) == 0))) {
	    LOG_ERROR("the capture holds %" PRIu64 " positions: %" PRIu32
		      " cannot run within the context of %u, "
		      "or the break's layer %u does not run",
		      captured, run.positions, MODEL_CONTEXT_WITHOUT_CANDIDATES,
		      TEST_BREAK_LAYER);
	    status = STATUS_INVALID_ARGUMENT;
    }
    (void)printf("model_v41_teacher_forced_card_test: %" PRIu32 " positions, layers run %#" PRIx64 " (asked %#" PRIx64
                 "), %s%s\n", run.positions, layers, options.layers, options.free_running ? "free" : "injected",
                 options.broken ? ", the carried pre of the break's layer as zeros" : "");
    status = status != STATUS_OK ?
		     status :
		     load_run(&load, &options, layers, &run, &does_not_fit);
    if (status == STATUS_OK &&
	!conditions_write(conditions, &options, &load, layers)) {
	    LOG_ERROR("the conditions of the run do not fit %u bytes",
		      CONDITIONS_CAPACITY);
	    status = STATUS_INVALID_ARGUMENT;
    }
    run.capture = capture;
    run.program = load.program;
    run.engram = load.engram;
    status = status != STATUS_OK ? status : test_teacher_forced_write(&run);
    load_release(&load);
    test_capture_close(capture);
#if TIMING_HOOKS_ENABLED
    hooks_report(stderr);
#endif
    (void)printf("model_v41_teacher_forced_card_test %s: %s\n",
		 status == STATUS_OK ? "wrote a complete "
				       "folder" :
		 does_not_fit ? "FAILED, the layers do not fit the cards" :
				"FAILED",
		 status_text(status));
    return status == STATUS_OK ? EXIT_SUCCESS :
	   does_not_fit	       ? TEST_EXIT_DOES_NOT_FIT :
				 EXIT_FAILURE;
}
