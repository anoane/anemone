/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * step_expert_pause_test: the step driver's expert pauses (source/step.c, step.h, model_seam.h) on the backend in
 * host memory, whose fill kernel a stand-in turns into the two layers of one card:
 *     a router writes the job table its script gives, a card expert by its offsets, a host expert by its tag;
 *     the experts copy the bytes of each live job's three matrices out of the arena, where the test reads them;
 *     the sampler answers each row.
 * Both layers keep experts in host memory (FIRST_HOST_EXPERTS and SECOND_HOST_EXPERTS) and share the card's slots.
 * The driver runs on that backend with its downloads held back as backend_seam.h allows: their bytes reach staging
 * memory only once a wait reports them finished (until then POISON), so a job table read before its wait fails.
 *
 *   - Steps: every live job's matrices are read where its expert's bytes are, from the card or uploaded into a slot,
 *     so the same kernels read the same bytes; an expert a slot holds is not uploaded again, by the same pause or a
 *     later one; a slot no job of the pause took goes next, in turn; the job table goes up again only where a tag
 *     was in it; the uploads come between the router and the experts.
 *   - Refusals: a job of a broken tag (off the tags' step, beyond the layer's, matrices apart, partly a tag), which
 *     fails the step and the driver; every rule of create for pauses, once, slots over the areas of the card among
 *     them (the hand-offs' and the n-gram rows': model_v41_host_experts_test); every refusal of the attach, including
 *     more than STEP_MOST_PINNED_EXPERT_BYTES of experts and staging memory refused; a step before the
 *     attach, nothing queued.
 * A failure is one line on the standard output: the case, and what was expected of it.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/backend_records.h"
#include "log_expectation.h"
#include "anemone/hooks.h"
#include "anemone/step.h"
#include "host_memory_backend.h"
#include "prepare_common.h"

#define ARENA_BYTES 131072u                            /* the limit case's slots fit it too */
#define STEP_OFFSET 0u
#define SESSION_OFFSET 256u
#define RESULT_OFFSET 512u
#define FIRST_JOBS_OFFSET 1024u
#define SECOND_JOBS_OFFSET 2048u
#define CARD_EXPERTS_OFFSET 4096u
#define SLOTS_OFFSET 8192u
#define OUTPUT_OFFSET 16384u                           /* job j of layer l's matrices at (l SLOTS + j) EXPERT_BYTES */
#define MATRIX_BYTES 64ull
#define MATRICES EXPERT_PAUSE_MATRICES
#define EXPERT_BYTES (MATRICES * MATRIX_BYTES)
#define SLOT_BYTES 256u
#define GATE_AT 0u                                     /* w1, w2, w3 in the order of their names: gate, down, up */
#define DOWN_AT MATRIX_BYTES
#define UP_AT (DOWN_AT + MATRIX_BYTES)
#define JOBS_PER_ROW 3u
#define ROW_CAPACITY 2u
#define SLOTS 6u                                       /* JOBS_PER_ROW times ROW_CAPACITY: a step's jobs at most */
#define CARD_EXPERTS 4u
#define FIRST_HOST_EXPERTS 8u
#define SECOND_HOST_EXPERTS 4u
#define HOST_EXPERTS (FIRST_HOST_EXPERTS + SECOND_HOST_EXPERTS)
#define HOST_STRIDE 320u                               /* apart in the host arena, not in the order of the slots */
#define HOST_FIRST_OFFSET 128u
#define HOST_ARENA_BYTES (HOST_FIRST_OFFSET + (HOST_EXPERTS - 1u) * HOST_STRIDE + EXPERT_BYTES)  /* to the last */
#define HOST 100u                                      /* a script's expert from HOST on: host expert e - HOST */
#define LAYERS 2u
#define PROGRAMS 2u
#define LAUNCHES 5u
#define SEGMENTS 5u
#define CONTEXT 64u
#define WAIT_LIMIT 1000000000ull
#define CLOSE_LIMIT 1000000000ull
#define CAPTURE_CAPACITY 4096u
#define PATTERN_EXPERT 37u
#define PATTERN_MATRIX 101u
#define PATTERN_BYTE 3u
#define OFF_STEP 8u                                    /* a tag off the tags' step */
#define LARGE_EXPERT_BYTES 16384u                      /* the limit case: this many bytes an expert, */
#define LARGE_HOST_EXPERTS                                                \
	((uint32_t)(STEP_MOST_PINNED_EXPERT_BYTES / LARGE_EXPERT_BYTES) + \
	 1u) /* one more */
#define WIDE_UPLOADS 5u                                /* of six jobs, one held */
#define FIRST_LAYER_UPLOADS 2u                         /* the first step's: two of the first layer's experts */
#define POISON 0xA5u                                   /* a held-back download's bytes until its wait */
#define MOST_HELD_BACK BACKEND_MAXIMUM_UPLOADS_IN_FLIGHT
#define STEP_AREA_BYTES \
	(STEP_BLOCK_BYTES_VALUE + ROW_CAPACITY * STEP_ROW_BYTES_VALUE)

_Static_assert(SLOTS == JOBS_PER_ROW * ROW_CAPACITY, "a slot for each job of the largest step");

enum operation { OPERATION_ROUTER = 1, OPERATION_EXPERTS = 2, OPERATION_SAMPLE = 3 };
enum tag_break { TAG_SOUND, TAG_OFF_STEP, TAG_BEYOND, TAG_APART, TAG_PART, TAG_BREAKS };

/* What a router writes at the next step: per layer its jobs' experts (below CARD_EXPERTS on the card, from HOST on in
 * host memory as numbered by its pause) and each job's rows. */
struct script {
    uint32_t job_count[LAYERS];
    uint32_t experts[LAYERS][SLOTS];
    uint32_t rows[LAYERS][SLOTS];
};

struct fixture {
	struct kernel_launch launches[PROGRAMS][LAUNCHES];
	struct segment_record segments[PROGRAMS][SEGMENTS];
	struct program_card_record program_cards[PROGRAMS][1];
	struct expert_pause pauses[PROGRAMS][LAYERS];
	struct execution_program programs[PROGRAMS];
	const struct execution_program *pointers[PROGRAMS];
};

static int failures = 0;
static const char *context_text = "";
static struct device_card *card;
static uint64_t host_offsets[HOST_EXPERTS];
static unsigned char host_arena[HOST_ARENA_BYTES];
static uint64_t other_offsets[HOST_EXPERTS];
static struct script script;
static enum tag_break tag_break = TAG_SOUND;
static char capture_path[] = "/tmp/anemone_step_expert_pause_test_XXXXXX";
static int capture_descriptor = -1;
static int error_stream_descriptor = -1;
static const uint64_t matrix_at[MATRICES] = {GATE_AT, UP_AT, DOWN_AT};   /* gate, up, down as a job holds them */
static struct backend_operations
	holding_backend; /* the backend, downloads held back */
static struct held_back { void *destination; unsigned char *bytes; uint64_t byte_count, ticket; } held[MOST_HELD_BACK];
static uint32_t held_count;

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
           "the error stream is pointed at the temporary file");
}

/* What was logged since capture_begin: error lines, at least one holding words. */
static void capture_end_expecting(const char *words)
{
    char text[CAPTURE_CAPACITY];
    const bool is_open = capture_descriptor >= 0 && error_stream_descriptor >= 0 && fflush(stderr) == 0;
    const ssize_t byte_count = is_open ? pread(capture_descriptor, text, sizeof(text) - 1, 0) : -1;
    expect(is_open && dup2(error_stream_descriptor, STDERR_FILENO) == STDERR_FILENO && byte_count >= 0,
           "the temporary file is read back and the error stream restored");
    text[byte_count >= 0 ? (size_t)byte_count : 0] = '\0';
    if (!test_log_is_expected(text, words, false)) {
        (void)printf("FAILED: %s: expected an error line with \"%s\", got \"%s\"\n", context_text, words, text);
        failures++;
    }
}

/* Byte k of matrix m (gate, up, down) of an expert: card expert e is e, host expert h is HOST + h. */
static unsigned char pattern_of(uint32_t expert, uint32_t matrix, uint32_t byte)
{
    return (unsigned char)(expert * PATTERN_EXPERT + matrix * PATTERN_MATRIX + byte * PATTERN_BYTE + 1u);
}

/* ---- the backend, downloads held back ---- */

/* A download as the backend's, its bytes kept aside and POISON in their place until a wait reports it finished. */
MUST_CHECK static result_status
held_download_begin(struct device_card *opened, uint64_t arena_byte_offset,
		    void *page_locked_destination, uint64_t byte_count,
		    uint64_t *ticket_out)
{
	const result_status status = test_host_memory_backend.download_begin(
		opened, arena_byte_offset, page_locked_destination, byte_count,
		ticket_out);
	unsigned char *const bytes =
		status == STATUS_OK && held_count < MOST_HELD_BACK ?
			malloc(byte_count) :
			NULL;
	expect(status != STATUS_OK || bytes != NULL, "a download is held back");
	if (bytes != NULL) {
		memcpy(bytes, page_locked_destination, byte_count);
		memset(page_locked_destination, POISON, byte_count);
		held[held_count++] =
			(struct held_back){ page_locked_destination, bytes,
					    byte_count, *ticket_out };
	}
    return status;
}

/* The backend's wait; the downloads it reports finished deliver their bytes. */
MUST_CHECK static result_status
held_upload_wait(struct device_card *opened, uint64_t ticket,
		 uint64_t wait_limit_nanoseconds)
{
	const result_status status = test_host_memory_backend.upload_wait(
		opened, ticket, wait_limit_nanoseconds);
	for (uint32_t index = 0; index < held_count && status == STATUS_OK;) {
		if (held[index].ticket <= ticket) {
			memcpy(held[index].destination, held[index].bytes,
			       held[index].byte_count);
			free(held[index].bytes);
			held[index] = held[--held_count];
		} else {
			index++;
		}
	}
    return status;
}

/* ---- the stand-ins ---- */

/* Job index of layer's table: a card expert's offsets, or a host expert's tag, numbered among the layer's. */
static void job_write(unsigned char *arena, uint64_t jobs_offset, uint32_t layer, uint32_t index, uint32_t expert,
                      uint32_t rows)
{
	struct expert_job_record job = { .row_count = rows };
	const uint32_t tag =
		expert - HOST - (layer == 0 ? 0 : FIRST_HOST_EXPERTS);
	uint64_t offsets[MATRICES];
	for (uint32_t matrix = 0; matrix < MATRICES; matrix++) {
		offsets[matrix] =
			expert < HOST ?
				CARD_EXPERTS_OFFSET + expert * EXPERT_BYTES +
					matrix_at[matrix] :
				SLOTS_OFFSET +
					(uint64_t)tag *
						LINEAR_OFFSET_ALIGNMENT_BYTES;
	}
    for (uint32_t matrix = 0; matrix < MATRICES && index == 0 && expert >= HOST; matrix++) {
	    offsets[matrix] =
		    tag_break == TAG_BEYOND ?
			    SLOTS_OFFSET +
				    FIRST_HOST_EXPERTS *
					    LINEAR_OFFSET_ALIGNMENT_BYTES :
			    offsets[matrix] +
				    (tag_break == TAG_OFF_STEP ? OFF_STEP : 0);
    }
    if (index == 0 && expert >= HOST) {
	    offsets[1] +=
		    tag_break == TAG_APART ? LINEAR_OFFSET_ALIGNMENT_BYTES : 0;
	    offsets[0] = tag_break == TAG_PART ? CARD_EXPERTS_OFFSET :
						 offsets[0];
    }
    job.gate_offset = offsets[0];
    job.up_offset = offsets[1];
    job.down_offset = offsets[MATRICES - 1u];
    for (uint32_t row = 0; row < rows; row++) {
        job.pair_index[row] = row * JOBS_PER_ROW + index % JOBS_PER_ROW;
        job.route_weight[row] = 1.0f;
    }
    memcpy(arena + jobs_offset + (uint64_t)index * EXPERT_JOB_BYTES, &job,
	   sizeof(job));
}

/* The fill kernel as the card's kernels: the record's value is the operation, its offset the layer's jobs, its word
 * count the layer; the grid the program's rows. */
static bool card_stand_in(const struct kernel_launch *launch,
			  uint32_t card_index, unsigned char *arena,
			  uint64_t arena_byte_count, void *context)
{
	struct fill_record record;
	struct step_block block;
	memcpy(&record, launch->argument, sizeof(record));
	memcpy(&block, arena + STEP_OFFSET, sizeof(block));
	(void)context;
	const uint32_t layer = (uint32_t)record.word_count,
		       program_rows = launch->grid[0];
	if (card_index != 0 || arena_byte_count != ARENA_BYTES ||
	    layer >= LAYERS || block.row_count > program_rows) {
		return false;
	}
    if (record.value == OPERATION_ROUTER) {
	    memset(arena + record.arena_byte_offset, 0,
		   (size_t)JOBS_PER_ROW * program_rows * EXPERT_JOB_BYTES);
	    for (uint32_t index = 0; index < script.job_count[layer]; index++) {
		    job_write(arena, record.arena_byte_offset, layer, index,
			      script.experts[layer][index],
			      script.rows[layer][index]);
	    }
    } else if (record.value == OPERATION_EXPERTS) {
        for (uint32_t index = 0; index < JOBS_PER_ROW * block.row_count; index++) {
		struct expert_job_record job;
		memcpy(&job,
		       arena + record.arena_byte_offset +
			       (uint64_t)index * EXPERT_JOB_BYTES,
		       sizeof(job));
		const uint64_t offsets[MATRICES] = { job.gate_offset,
						     job.up_offset,
						     job.down_offset };
		for (uint32_t matrix = 0;
		     matrix < MATRICES && job.row_count != 0; matrix++) {
			if (offsets[matrix] > arena_byte_count - MATRIX_BYTES) {
				return false;
			}
			memcpy(arena + OUTPUT_OFFSET +
				       ((uint64_t)layer * SLOTS + index) *
					       EXPERT_BYTES +
				       matrix * MATRIX_BYTES,
			       arena + offsets[matrix], MATRIX_BYTES);
		}
        }
    } else {
	    const struct step_result answer = { .sequence = block.sequence,
						.row_count = block.row_count };
	    const struct step_row_result result = { .token_id = 0,
						    .logit = 0.0f };
	    memcpy(arena + RESULT_OFFSET, &answer, sizeof(answer));
	    for (uint32_t row = 0; row < block.row_count; row++) {
		    memcpy(arena + RESULT_OFFSET + STEP_RESULT_BYTES +
				   (size_t)row * STEP_ROW_RESULT_BYTES,
			   &result, sizeof(result));
	    }
    }
    return true;
}

/* ---- the fixture ---- */

static void fixture_link(struct fixture *fixture)
{
    for (uint32_t program = 0; program < PROGRAMS; program++) {
        fixture->program_cards[program][0].launches = fixture->launches[program];
        fixture->programs[program].cards = fixture->program_cards[program];
        fixture->programs[program].segments = fixture->segments[program];
        fixture->programs[program].expert_pauses = fixture->pauses[program];
        fixture->programs[program].host_expert_offsets = host_offsets;
        fixture->pointers[program] = &fixture->programs[program];
    }
}

/* Programs of 1 and ROW_CAPACITY rows: router, experts of each layer, then the sampler, a pause behind each router. */
static void fixture_build(struct fixture *fixture, uint32_t kernel)
{
    static const uint32_t operations[LAUNCHES] = {OPERATION_ROUTER, OPERATION_EXPERTS, OPERATION_ROUTER,
                                                  OPERATION_EXPERTS, OPERATION_SAMPLE};
    static const uint64_t jobs_offsets[LAYERS] = {FIRST_JOBS_OFFSET, SECOND_JOBS_OFFSET};
    const struct model_limits limits = { .context_token_limit = CONTEXT,
					 .row_capacity = ROW_CAPACITY,
					 .session_capacity = 1,
					 .state_width = STATE_WIDTH_16 };
    memset(fixture, 0, sizeof(*fixture));
    for (uint32_t program = 0; program < PROGRAMS; program++) {
        const uint32_t rows = program == 0 ? 1u : ROW_CAPACITY;
        for (uint32_t launch = 0; launch < LAUNCHES; launch++) {
            const uint32_t layer = launch / LAYERS < LAYERS ? launch / LAYERS : 0;
	    const struct fill_record record = { .arena_byte_offset =
							jobs_offsets[layer],
						.word_count = layer,
						.value = operations[launch] };
	    prepare_launch_fill(&fixture->launches[program][launch], kernel,
				rows, 1, 1, 0, &record, sizeof(record));
	    fixture->segments[program][launch] = (struct segment_record){
		    .card = 0,
		    .stage = 0,
		    .layer = launch + 1u < LAUNCHES ? layer : SEGMENT_NO_LAYER,
		    .first_launch = launch,
		    .launch_count = 1,
		    .pause = operations[launch] == OPERATION_ROUTER ?
				     SEGMENT_PAUSE_EXPERT_UPLOADS :
				     0
	    };
	}
        for (uint32_t layer = 0; layer < LAYERS; layer++) {
		fixture->pauses[program][layer] = (struct expert_pause){
			jobs_offsets[layer],
			SLOTS_OFFSET,
			SLOT_BYTES,
			EXPERT_BYTES,
			{ GATE_AT, UP_AT, DOWN_AT },
			layer == 0 ? 0 : FIRST_HOST_EXPERTS,
			layer == 0 ? FIRST_HOST_EXPERTS : SECOND_HOST_EXPERTS,
			JOBS_PER_ROW,
			SLOTS,
			0,
			layer * LAYERS
		};
	}
	fixture->program_cards[program][0] = (struct program_card_record){
		.arena_byte_count = ARENA_BYTES,
		.step_offset = STEP_OFFSET,
		.session_offset = SESSION_OFFSET,
		.result_offset = RESULT_OFFSET,
		.logits_offset = PROGRAM_NOWHERE,
		.round_offset = PROGRAM_NOWHERE,
		.streams_offset = PROGRAM_NOWHERE,
		.carried_pre_offset = PROGRAM_NOWHERE,
		.engram_rows_offset = PROGRAM_NOWHERE,
		.launch_count = LAUNCHES
	};
	fixture->programs[program] =
		(struct execution_program){ .limits = limits,
					    .card_count = 1,
					    .segment_count = SEGMENTS,
					    .draft_segment_first = SEGMENTS,
					    .row_count = rows,
					    .result_card = 0,
					    .expert_pause_count = LAYERS,
					    .host_expert_count = HOST_EXPERTS };
    }
    fixture_link(fixture);
}

/* Opens the card with the card experts' bytes in place; the host arena holds the host experts' at host_offsets. */
static uint32_t card_open(void)
{
    uint32_t kernel = 0;
    void *staging = NULL;
    uint64_t ticket = 0;
    test_host_memory_backend_reset(1, ARENA_BYTES);
    test_host_memory_backend_set_arithmetic_class(0, ARITHMETIC_CLASS_SM_80);
    expect_status(test_host_memory_backend.card_open(0, ARENA_BYTES, &card),
		  STATUS_OK, "the card opens");
    expect_status(test_host_memory_backend.kernel_prepare(
			  card, KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
			  FILL_RECORD_BYTES, &kernel),
		  STATUS_OK, "the fill kernel");
    expect_status(test_host_memory_backend_set_stand_in(KERNEL_FILE_FILL,
							KERNEL_FILL_WORDS,
							card_stand_in, NULL),
		  STATUS_OK, "the stand-in");
    expect_status(test_host_memory_backend.staging_allocate(
			  CARD_EXPERTS * EXPERT_BYTES, &staging),
		  STATUS_OK, "staging for the card experts");
    for (uint32_t expert = 0; expert < HOST_EXPERTS; expert++) {
        host_offsets[expert] = HOST_FIRST_OFFSET + (uint64_t)(HOST_EXPERTS - 1u - expert) * HOST_STRIDE;
        for (uint32_t matrix = 0; matrix < MATRICES; matrix++) {
            for (uint32_t byte = 0; byte < MATRIX_BYTES; byte++) {
                host_arena[host_offsets[expert] + matrix_at[matrix] + byte] = pattern_of(HOST + expert, matrix, byte);
                if (expert < CARD_EXPERTS && staging != NULL) {
                    ((unsigned char *)staging)[expert * EXPERT_BYTES + matrix_at[matrix] + byte] =
                        pattern_of(expert, matrix, byte);
                }
            }
        }
    }
    expect(staging != NULL &&
		   test_host_memory_backend.upload_begin(
			   card, CARD_EXPERTS_OFFSET, staging,
			   CARD_EXPERTS * EXPERT_BYTES, &ticket) == STATUS_OK &&
		   test_host_memory_backend.upload_wait(
			   card, ticket, WAIT_LIMIT) == STATUS_OK,
	   "the card experts' bytes are on the card");
    test_host_memory_backend.staging_free(staging);
    tag_break = TAG_SOUND;
    memset(&script, 0, sizeof(script));
    return kernel;
}

static void card_close(void)
{
    test_host_memory_backend.card_close(card, CLOSE_LIMIT);
    card = NULL;
    for (; held_count != 0; held_count--) {             /* what no wait delivered, as a closed card's */
        free(held[held_count - 1u].bytes);
    }
}

static struct step_driver *driver_of(const struct fixture *fixture,
				     bool is_attached)
{
	struct step_driver *driver = NULL;
	expect_status(step_driver_create(&holding_backend, &card,
					 fixture->pointers, PROGRAMS, NULL,
					 WAIT_LIMIT, &driver),
		      STATUS_OK, "the driver is made");
	if (is_attached) {
		expect_status(step_host_experts_attach(driver, host_arena,
						       HOST_ARENA_BYTES),
			      STATUS_OK, "the host experts are attached");
	}
    return driver;
}

/* ---- steps ---- */

static uint32_t step_position;

/* A step of row_count rows of the script's routing: every live job read its expert's bytes; uploads[l] experts of
 * layer l went up into the slots, between its router and its experts, and none elsewhere; its job table went up
 * again there where a tag was in it. */
static void step_expecting(struct step_driver *driver, uint32_t row_count,
			   const uint32_t uploads[LAYERS])
{
    static const uint64_t jobs_offsets[LAYERS] = {FIRST_JOBS_OFFSET, SECOND_JOBS_OFFSET};
    struct step_row_record rows[ROW_CAPACITY];
    struct step_row_result results[ROW_CAPACITY];
    const uint64_t work_before = test_host_memory_backend_work_count();
    uint32_t seen_uploads[LAYERS] = {0}, table_uploads[LAYERS] = {0}, launches = 0, stray_copies = 0;
    for (uint32_t row = 0; row < row_count; row++) {
	    rows[row] = (struct step_row_record){ (int32_t)step_position++, 1,
						  0, STEP_ROW_LOGITS };
    }
    expect_status(step_run_entry(driver, rows, row_count, results), STATUS_OK,
		  "a step runs");
    for (uint64_t index = work_before; index < test_host_memory_backend_work_count(); index++) {
        struct test_host_memory_backend_work work = {.kind = 0};
        expect(test_host_memory_backend_work_at(index, &work), "the work is logged");
        const uint32_t layer = launches / LAYERS;                 /* behind its router, before its experts: odd */
        const bool is_paused = launches % LAYERS == 1u && layer < LAYERS;
        const bool is_upload = work.kind == TEST_HOST_MEMORY_BACKEND_UPLOAD;
        if (work.kind == TEST_HOST_MEMORY_BACKEND_LAUNCH) {
            launches++;
        } else if (is_upload && work.arena_byte_offset >= SLOTS_OFFSET && work.byte_count == EXPERT_BYTES) {
            seen_uploads[is_paused ? layer : 0] += is_paused ? 1u : 0u;
            stray_copies += is_paused ? 0 : 1u;
        } else if (is_upload && (work.arena_byte_offset == FIRST_JOBS_OFFSET ||
                                 work.arena_byte_offset == SECOND_JOBS_OFFSET)) {
            const bool is_its_own = is_paused && work.arena_byte_offset == jobs_offsets[layer];
            table_uploads[is_paused ? layer : 0] += is_its_own ? 1u : 0u;
            stray_copies += is_its_own ? 0 : 1u;
        }
    }
    expect(stray_copies == 0, "the experts and the job tables go up in the pause of their layer only");
    const unsigned char *const arena = test_host_memory_backend_arena_of(card);
    for (uint32_t layer = 0; layer < LAYERS && arena != NULL; layer++) {
        bool is_read = true, has_tag = false;
        for (uint32_t index = 0; index < script.job_count[layer]; index++) {
            const uint32_t expert = script.experts[layer][index];
            has_tag = has_tag || expert >= HOST;
            for (uint32_t matrix = 0; matrix < MATRICES; matrix++) {
                for (uint32_t byte = 0; byte < MATRIX_BYTES; byte++) {
                    is_read = is_read && arena[OUTPUT_OFFSET + ((uint64_t)layer * SLOTS + index) * EXPERT_BYTES +
                                               matrix * MATRIX_BYTES + byte] == pattern_of(expert, matrix, byte);
                }
            }
        }
        expect(is_read, "every live job read its expert's bytes, on the card or in a slot");
        expect(seen_uploads[layer] == uploads[layer], "the experts of the layer that went up into the slots");
        expect(table_uploads[layer] == (has_tag ? 1u : 0u), "the job table goes up again only where a tag was in it");
    }
}

static void script_set(uint32_t layer, uint32_t job_count, const uint32_t *experts)
{
    script.job_count[layer] = job_count;
    for (uint32_t index = 0; index < job_count; index++) {
        script.experts[layer][index] = experts[index];
        script.rows[layer][index] = 1;
    }
}

static void test_steps(void)
{
    static const uint32_t first_routes[] = {HOST + 2u, 1, HOST + 5u};
    static const uint32_t second_routes[] = {HOST + FIRST_HOST_EXPERTS + 1u};
    static const uint32_t again_routes[] = {HOST + 5u, HOST + 2u};
    static const uint32_t wide_routes[SLOTS] = {HOST, HOST + 1u, HOST + 3u, HOST + 4u, HOST + 2u, HOST + 6u};
    static const uint32_t card_routes[] = {0, CARD_EXPERTS - 1u};
    struct fixture fixture;
    context_text = "steps";
    fixture_build(&fixture, card_open());
    step_position = 0;
    struct step_driver *const driver = driver_of(&fixture, true);
    /* two host experts and a card expert in the first layer, one of the second's: slots 0, 1 and 2 */
    script_set(0, sizeof(first_routes) / sizeof(first_routes[0]), first_routes);
    script_set(1, 1, second_routes);
    step_expecting(driver, 1, (const uint32_t[LAYERS]){FIRST_LAYER_UPLOADS, 1u});
    /* the same experts again: the slots hold them, nothing goes up */
    script_set(0, sizeof(again_routes) / sizeof(again_routes[0]), again_routes);
    step_expecting(driver, 1, (const uint32_t[LAYERS]){0, 0});
    /* two rows, six host experts in the first layer: expert 2 held in slot 0, five go up into the slots after it in
     * turn (3, 4, 5, then 1 and 2 past the marked 0), the second layer on the card alone */
    script_set(0, SLOTS, wide_routes);
    script_set(1, sizeof(card_routes) / sizeof(card_routes[0]), card_routes);
    step_expecting(driver, ROW_CAPACITY, (const uint32_t[LAYERS]){WIDE_UPLOADS, 0});
    /* the second layer's expert lost its slot 2: it goes up again, into the next in turn */
    script_set(0, 1, card_routes);
    script_set(1, 1, second_routes);
    step_expecting(driver, 1, (const uint32_t[LAYERS]){0, 1u});
    step_driver_destroy(driver);
    expect(test_host_memory_backend_staging_block_count() == 0, "destroy frees both staging blocks");
    card_close();
}

/* ---- refusals ---- */

static void test_broken_tags(void)
{
    static const uint32_t routes[] = {HOST + 3u, 0};
    static const char *const cases[TAG_BREAKS] = {"", "a tag off the step", "a tag beyond the layer's",
                                                  "a job's matrices apart", "a job partly a tag"};
    struct fixture fixture;
    struct step_row_result result;
    const struct step_row_record row = { 0, 1, 0, STEP_ROW_LOGITS };
    for (uint32_t index = TAG_OFF_STEP; index < TAG_BREAKS; index++) {
        context_text = cases[index];
        fixture_build(&fixture, card_open());
        tag_break = (enum tag_break)index;
        script_set(0, sizeof(routes) / sizeof(routes[0]), routes);
	struct step_driver *const driver = driver_of(&fixture, true);
	capture_begin();
	expect_status(step_run_entry(driver, &row, 1, &result),
		      STATUS_DEVICE_ERROR, "the step fails");
	capture_end_expecting("names neither an expert on card 0 nor one of the 8 in host memory");
	expect_status(step_run_entry(driver, &row, 1, &result),
		      STATUS_DEVICE_ERROR,
		      "the driver answers so from then on");
	step_driver_destroy(driver);
	card_close();
    }
}

/* create refuses a changed copy of the fixture, with one error line, and leaves its output alone. */
static void expect_refused(const struct fixture *fixture, const char *what)
{
	struct step_driver *driver =
		(void *)&failures; /* an address no create gives */
	capture_begin();
	expect_status(step_driver_create(&holding_backend, &card,
					 fixture->pointers, PROGRAMS, NULL,
					 WAIT_LIMIT, &driver),
		      STATUS_INVALID_ARGUMENT, what);
	capture_end_expecting("step driver: refused");
	expect(driver == (void *)&failures &&
		       test_host_memory_backend_staging_block_count() == 0,
	       "nothing is given or kept");
}

static void test_create_refusals(void)
{
    struct fixture base;
    struct fixture changed;
    context_text = "create's refusals";
    fixture_build(&base, card_open());
#define CHANGED(change, what)                                           \
    do {                                                                \
	    changed = base;                                             \
	    fixture_link(&changed);                                     \
	    for (uint32_t program = 0; program < PROGRAMS; program++) { \
		    struct expert_pause *const pause =                  \
			    &changed.pauses[program][1];                \
		    struct segment_record *const segments =             \
			    changed.segments[program];                  \
		    struct execution_program *const made =              \
			    &changed.programs[program];                 \
		    struct program_card_record *const area =            \
			    &changed.program_cards[program][0];         \
		    (void)pause;                                        \
		    (void)segments;                                     \
		    (void)made;                                         \
		    (void)area;                                         \
		    change;                                             \
	    }                                                           \
	    expect_refused(&changed, what);                             \
    } while (false)
    CHANGED(made->expert_pause_count = 1, "a paused segment without its pause");
    CHANGED(segments[LAYERS].pause = 0, "a pause behind a segment that does not pause");
    CHANGED(segments[LAYERS].pause = 0;
	    segments[LAYERS + 1u].pause = SEGMENT_PAUSE_EXPERT_UPLOADS,
	    "a pause behind a segment that does not pause, the next pausing in its place");
    CHANGED(pause->after_segment = 0, "pauses out of order");
    CHANGED(pause->card = 1, "a pause on a card the program does not have");
    CHANGED(pause->slot_count = SLOTS - 1u; changed.pauses[program][0].slot_count = SLOTS - 1u,
            "fewer slots than the jobs of a step");
    CHANGED(pause->slot_count = SLOTS + 1u, "a card's slots unlike in its other pause");
    CHANGED(pause->slots_offset = SLOTS_OFFSET + SLOT_BYTES, "a card's slots elsewhere in its other pause");
    CHANGED(pause->expert_byte_count = SLOT_BYTES + 1u, "an expert larger than a slot");
    CHANGED(pause->matrix_offsets[1] = EXPERT_BYTES, "a matrix beyond the expert");
    CHANGED(pause->matrix_offsets[MATRICES - 1u] = 1u, "a matrix off the alignment");
    CHANGED(pause->jobs_offset = SLOTS_OFFSET, "the jobs among the slots");
    CHANGED(pause->jobs_offset = ARENA_BYTES - EXPERT_JOB_BYTES,
	    "jobs that end behind the arena");
    CHANGED(pause->slots_offset = ARENA_BYTES - SLOT_BYTES; changed.pauses[program][0].slots_offset =
                ARENA_BYTES - SLOT_BYTES, "slots that end behind the arena");
    CHANGED(pause->host_expert_count = 0, "a pause of no host expert");
    CHANGED(pause->host_expert_count = SECOND_HOST_EXPERTS + 1u, "host experts beyond the program's");
    CHANGED(pause->first_host_expert = FIRST_HOST_EXPERTS - 1u, "pauses whose host experts overlap");
    CHANGED(area->step_offset = SLOTS_OFFSET + 1u - STEP_AREA_BYTES, "the step area over the first slot's first byte");
    CHANGED(area->session_offset = SLOTS_OFFSET + SLOTS * SLOT_BYTES - 1u,
            "the session settings over the last slot's last byte");
    CHANGED(area->result_offset = SLOTS_OFFSET + SLOT_BYTES, "the result area over a slot");
    CHANGED(pause->jobs_per_row = 0, "no job a row");
    CHANGED(made->host_expert_offsets = NULL, "no host experts' offsets");
    CHANGED(if (program == 1) { pause->first_host_expert = 1; }, "programs of other pauses");
    memcpy(other_offsets, host_offsets, sizeof(other_offsets));
    other_offsets[HOST_EXPERTS - 1u]++;
    CHANGED(if (program == 1) { made->host_expert_offsets = other_offsets; }, "programs of other host experts");
#undef CHANGED
    card_close();
}

static void test_attach_refusals(void)
{
    struct fixture fixture;
    struct step_row_result result;
    const struct step_row_record row = { 0, 1, 0, STEP_ROW_LOGITS };
    context_text = "the attach's refusals";
    fixture_build(&fixture, card_open());
    struct step_driver *driver = driver_of(&fixture, false);
    const uint64_t work_before = test_host_memory_backend_work_count();
    capture_begin();
    expect_status(step_run_entry(driver, &row, 1, &result),
		  STATUS_INVALID_ARGUMENT, "a step before the attach");
    capture_end_expecting("attached host experts");
    expect(test_host_memory_backend_work_count() == work_before, "nothing is queued");
    static const struct { bool has_arena; uint64_t byte_count; const char *what; } cases[] = {
        {false, HOST_ARENA_BYTES, "no host arena"}, {true, HOST_ARENA_BYTES - 1u, "an expert beyond the arena"}};
    for (uint32_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        capture_begin();
	expect_status(step_host_experts_attach(
			      driver,
			      cases[index].has_arena ? host_arena : NULL,
			      cases[index].byte_count),
		      STATUS_INVALID_ARGUMENT, cases[index].what);
	capture_end_expecting("host experts refused");
    }
    test_host_memory_backend_fail_staging_allocation(1);
    expect_status(step_host_experts_attach(driver, host_arena,
					   HOST_ARENA_BYTES),
		  STATUS_OUT_OF_MEMORY, "staging memory refused");
    expect(test_host_memory_backend_staging_block_count() == 1u, "only the driver's own block is left");
    expect_status(step_host_experts_attach(driver, host_arena,
					   HOST_ARENA_BYTES),
		  STATUS_OK, "the attach once staging memory is there");
    capture_begin();
    expect_status(step_host_experts_attach(driver, host_arena,
					   HOST_ARENA_BYTES),
		  STATUS_INVALID_ARGUMENT, "a second attach");
    capture_end_expecting("attached before");
    capture_begin();
    expect_status(step_host_experts_attach(NULL, host_arena, HOST_ARENA_BYTES),
		  STATUS_INVALID_ARGUMENT, "no driver");
    capture_end_expecting("no driver");
    step_driver_destroy(driver);
    /* one pause of one expert more than the limit holds, each of LARGE_EXPERT_BYTES, all at the arena's start */
    uint64_t *const large_offsets = calloc(LARGE_HOST_EXPERTS, sizeof(uint64_t));
    for (uint32_t program = 0; program < PROGRAMS && large_offsets != NULL; program++) {
	    fixture.pauses[program][0] =
		    (struct expert_pause){ FIRST_JOBS_OFFSET,
					   SLOTS_OFFSET,
					   LARGE_EXPERT_BYTES,
					   LARGE_EXPERT_BYTES,
					   { GATE_AT, UP_AT, DOWN_AT },
					   0,
					   LARGE_HOST_EXPERTS,
					   JOBS_PER_ROW,
					   SLOTS,
					   0,
					   0 };
	    fixture.programs[program].expert_pause_count = 1;
	    fixture.programs[program].host_expert_count = LARGE_HOST_EXPERTS;
	    fixture.programs[program].host_expert_offsets = large_offsets;
	    fixture.segments[program][LAYERS].pause = 0;
    }
    driver = large_offsets != NULL ? driver_of(&fixture, false) : NULL;
    capture_begin();
    expect_status(step_host_experts_attach(driver, host_arena,
					   LARGE_EXPERT_BYTES),
		  STATUS_OUT_OF_MEMORY, "more host experts than the limit");
    capture_end_expecting("or more than");
    expect(test_host_memory_backend_staging_block_count() == 1u, "nothing more is page-locked");
    step_driver_destroy(driver);
    free(large_offsets);
    /* a failed driver answers its failure */
    card_close();
    fixture_build(&fixture, card_open());
    tag_break = TAG_BEYOND;
    script_set(0, 1, (const uint32_t[]){HOST});
    driver = driver_of(&fixture, true);
    capture_begin();
    expect_status(step_run_entry(driver, &row, 1, &result), STATUS_DEVICE_ERROR,
		  "a step fails");
    capture_end_expecting("names neither");
    expect_status(step_host_experts_attach(driver, host_arena,
					   HOST_ARENA_BYTES),
		  STATUS_DEVICE_ERROR, "the attach of a failed driver");
    step_driver_destroy(driver);
    card_close();
}

int main(void)
{
    (void)setvbuf(stdout, NULL, _IOLBF, 0);             /* a failure's line survives a sanitizer's stop */
    capture_descriptor = mkstemp(capture_path);
    error_stream_descriptor = dup(STDERR_FILENO);
    holding_backend = test_host_memory_backend;
    holding_backend.download_begin = held_download_begin;
    holding_backend.upload_wait = held_upload_wait;
    if (capture_descriptor < 0 || error_stream_descriptor < 0 || unlink(capture_path) != 0) {
        (void)printf("FAILED: the temporary file for the error stream\n");
        return EXIT_FAILURE;
    }
    test_steps();
    test_broken_tags();
    test_create_refusals();
    test_attach_refusals();
    (void)close(capture_descriptor);
    (void)close(error_stream_descriptor);
    if (failures != 0) {
        (void)printf("step_expert_pause_test: %d failures\n", failures);
        return EXIT_FAILURE;
    }
    (void)printf("step_expert_pause_test: passed\n");
    return EXIT_SUCCESS;
}
