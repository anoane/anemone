/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * model_v41_card_test: what the model's plan puts on the cards, on the real cards, and its frequency words against
 * the reference's own (first_token_plan.md 2.10, 3.2: the table gate).
 *
 *     model_v41_card_test [--break-word | --break-state] [<capture folder>]
 *
 *   - Words: with a capture folder of tools/oracle/stepwise_capture.py --frequencies, its records
 *     frequencies.theta_10000.words and frequencies.theta_160000.words (the tensor torch.outer receives, formed on
 *     the card as the reference builds its model) must be v41_frequency_words bit for bit; a word that
 *     differs is printed with both bits and FAILS the test (model_v41.c holds the captured words as a table: the
 *     host's expression was a place off in two). Without a folder, as `make card_tests` runs every card test, it
 *     says no words were compared.
 *   - State: on every card, arena_initialise of a plan of one card, behind 64 KiB that stand for weights: 40 MiB of
 *     state over bytes the card held before, its zeros written by the fill kernel on the card and its image (a
 *     pattern) uploaded; the card's digest of the whole state must be the host's, and the state downloaded back the
 *     image then zeros, byte for byte. First the same plan on the card opened 4 KiB short: refused before anything
 *     is written (compute-sanitizer sees any write past the arena).
 *   - Shown failing: --break-word changes the last bit of host word 17 of theta 1 before the comparison (needs a
 *     capture folder), --break-state one zero byte of each card's state after arena_initialise. A break run passes
 *     only when its check reported exactly that change (that word differing by that bit and every other word equal;
 *     on every card that byte and no other) and nothing else failed.
 * A failure is one line on the standard error stream; the exit status is 0 only when every check held as asked.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "anemone/hooks.h"
#include "anemone/status.h"
#include "capture_reader.h"
#include "model_v41.h"

#define TEST_TIME_LIMIT_SECONDS 900             /* the alarm ends the test here, whatever the driver does */
#define WAIT_LIMIT_NANOSECONDS 120000000000ull  /* two minutes: far above 40 MiB over any link */
#define WEIGHTS_BYTES 65536ull
#define STATE_BYTES (40ull << 20u)              /* far past the widest grid of a fill */
#define IMAGE_BYTES 102400ull
#define IMAGE_SEED 0x9E3779B9u
#define IMAGE_SHIFT 13u
#define GARBAGE_BYTE 0xABu
#define BROKEN_STATE_OFFSET (IMAGE_BYTES + 777777ull)    /* a byte of the zeros */
#define SHORT_ARENA_BYTES 4096ull
#define BROKEN_WORD_THETA 1u
#define BROKEN_WORD_INDEX 17u                   /* inside the YaRN band */
#define WORD_BITS_FLIP 1u
#define THETA_NAMES {"frequencies.theta_10000.words", "frequencies.theta_160000.words"}
#define MOST_ARGUMENTS 3

enum break_kind { BREAK_NONE, BREAK_WORD, BREAK_STATE };

static const struct backend_operations *const backend = &backend_cuda_value;
static int failures = 0;                        /* checks that failed, the change a break made aside */
static int reported = 0;                        /* the change a break made, reported by its check */
static enum break_kind break_kind = BREAK_NONE;

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)fprintf(stderr, "FAILED: %s\n", what);
        failures++;
    }
}

static void expect_status(result_status status, result_status expected,
			  const char *what)
{
    if (status != expected) {
	    (void)fprintf(stderr, "FAILED: %s: %s, not %s\n", what,
			  status_text(status), status_text(expected));
	    failures++;
    }
}

/* The table gate: every word of both thetas against the capture's, bit for bit. */
static void test_words(const char *folder)
{
	static const char *const names[MODEL_THETAS] = THETA_NAMES;
	const uint64_t dimensions[] = { MODEL_FREQUENCIES };
	float words[MODEL_THETAS][MODEL_FREQUENCIES];
	float captured[MODEL_FREQUENCIES];
	struct test_capture *capture = NULL;
	uint32_t differing = 0;
	v41_frequency_words(words);
	if (break_kind == BREAK_WORD) {
		uint32_t bits = 0;
		memcpy(&bits, &words[BROKEN_WORD_THETA][BROKEN_WORD_INDEX],
		       sizeof(bits));
		bits ^= WORD_BITS_FLIP;
		memcpy(&words[BROKEN_WORD_THETA][BROKEN_WORD_INDEX], &bits,
		       sizeof(bits));
	}
	expect_status(test_capture_open(folder, &capture), STATUS_OK,
		      "the capture folder opens");
	for (uint32_t theta = 0; capture != NULL && theta < MODEL_THETAS;
	     theta++) {
		const result_status status = test_capture_read(
			capture, names[theta], TEST_CAPTURE_FLOAT32, dimensions,
			1, 0, MODEL_FREQUENCIES, captured, sizeof(captured));
		expect_status(
			status, STATUS_OK,
			"the capture holds the 32 float32 words of the theta");
		for (uint32_t index = 0;
		     status == STATUS_OK && index < MODEL_FREQUENCIES;
		     index++) {
			uint32_t host_bits = 0, reference_bits = 0;
			memcpy(&host_bits, &words[theta][index],
			       sizeof(host_bits));
			memcpy(&reference_bits, &captured[index],
			       sizeof(reference_bits));
			const bool is_broken = break_kind == BREAK_WORD &&
					       theta == BROKEN_WORD_THETA &&
					       index == BROKEN_WORD_INDEX;
			if (host_bits != reference_bits) {
				(void)fprintf(stderr,
					      "%s %s word %" PRIu32
					      ": the host's %08" PRIx32
					      ", the reference's %08" PRIx32
					      "\n",
					      is_broken ? "broken on purpose:" :
							  "FAILED:",
					      names[theta], index, host_bits,
					      reference_bits);
				differing++;
			}
			/* a break is shown only as that word differing by that bit; anything else is a failure of the gate */
			reported += is_broken && (host_bits ^ reference_bits) ==
							    WORD_BITS_FLIP ?
					    1 :
					    0;
			failures +=
				is_broken ?
					((host_bits ^ reference_bits) !=
							 WORD_BITS_FLIP ?
						 1 :
						 0) :
					(host_bits != reference_bits ? 1 : 0);
		}
	}
    test_capture_close(capture);
    (void)printf("frequency words: %" PRIu32
		 " of %u differ from the reference's\n",
		 differing, MODEL_THETAS * MODEL_FREQUENCIES);
}

/* The state of one card written by arena_initialise and read back, after the plan was refused on the card opened
 * short of its arena. */
static void test_card(uint32_t card_index, unsigned char *image, unsigned char *readback)
{
	struct model_plan_record *const plan = calloc(1, sizeof(*plan));
	struct device_card *card = NULL;
	uint64_t ticket = 0, differing = 0, first_differing = STATE_BYTES;
	bool broken_is_seen = false;
	if (plan == NULL) {
		expect(false, "memory for a plan");
		return;
	}
    plan->card_count = 1;
    plan->cards[0] =
	    (struct v41_card){ .state_offset = WEIGHTS_BYTES,
			       .arena_byte_count = WEIGHTS_BYTES + STATE_BYTES,
			       .image_byte_count = IMAGE_BYTES,
			       .image = image };
    expect_status(
	    backend->card_open(card_index,
			       WEIGHTS_BYTES + STATE_BYTES - SHORT_ARENA_BYTES,
			       &card),
	    STATUS_OK, "the card opens short of the plan's arena");
    expect_status(
	    card == NULL ? STATUS_DEVICE_ERROR :
			   v41_arena_initialise(plan, backend, &card,
						WAIT_LIMIT_NANOSECONDS),
	    STATUS_INVALID_ARGUMENT,
	    "an arena shorter than the plan's is refused before any write");
    backend->card_close(card, WAIT_LIMIT_NANOSECONDS);
    card = NULL;
    expect_status(backend->card_open(card_index, WEIGHTS_BYTES + STATE_BYTES,
				     &card),
		  STATUS_OK, "the card opens");
    if (card != NULL) {
        memset(readback, GARBAGE_BYTE, (size_t)STATE_BYTES);
	expect_status(backend->upload_begin(card, WEIGHTS_BYTES, readback,
					    STATE_BYTES, &ticket),
		      STATUS_OK, "other bytes go into the state first");
	expect_status(backend->upload_wait(card, ticket,
					   WAIT_LIMIT_NANOSECONDS),
		      STATUS_OK, "the other bytes arrived");
	expect_status(
		v41_arena_initialise(plan, backend, &card,
				     WAIT_LIMIT_NANOSECONDS),
		STATUS_OK,
		"arena_initialise writes the state and the card's digest is the host's");
	if (break_kind == BREAK_STATE) {
            readback[0] = GARBAGE_BYTE;
	    expect_status(backend->upload_begin(
				  card, WEIGHTS_BYTES + BROKEN_STATE_OFFSET,
				  readback, 1, &ticket),
			  STATUS_OK, "one byte of the state is changed");
	    expect_status(backend->upload_wait(card, ticket,
					       WAIT_LIMIT_NANOSECONDS),
			  STATUS_OK, "the changed byte arrived");
	}
	expect_status(backend->download_begin(card, WEIGHTS_BYTES, readback,
					      STATE_BYTES, &ticket),
		      STATUS_OK, "the state is read back");
	expect_status(backend->upload_wait(card, ticket,
					   WAIT_LIMIT_NANOSECONDS),
		      STATUS_OK, "the state arrived on the host");
	for (uint64_t index = 0; index < STATE_BYTES; index++) {
            const bool differs = readback[index] != (index < IMAGE_BYTES ? image[index] : 0);
            const bool is_broken = break_kind == BREAK_STATE && index == BROKEN_STATE_OFFSET;
            broken_is_seen = broken_is_seen || (is_broken && differs);
            first_differing = differs && !is_broken && differing == 0 ? index : first_differing;
            differing += differs && !is_broken ? 1u : 0u;
        }
    }
    if (differing != 0 || (break_kind == BREAK_STATE && !broken_is_seen)) {
        (void)fprintf(stderr, "FAILED: card %" PRIu32 ": %" PRIu64 " bytes of the state are not the image and zeros, "
                      "the first at %" PRIu64 "%s\n", card_index, differing, first_differing,
                      break_kind == BREAK_STATE && !broken_is_seen ? "; the byte broken on purpose was not seen" : "");
        failures++;
    }
    reported += broken_is_seen ? 1 : 0;
    backend->card_close(card, WAIT_LIMIT_NANOSECONDS);
    free(plan);
}

int main(int argument_count, char **arguments)
{
    const char *folder = NULL;
    uint32_t card_count = 0;
    void *staging = NULL;
    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    for (int index = 1; index < argument_count && argument_count <= MOST_ARGUMENTS; index++) {
        break_kind = strcmp(arguments[index], "--break-word") == 0 ? BREAK_WORD
                   : strcmp(arguments[index], "--break-state") == 0 ? BREAK_STATE : break_kind;
        folder = strncmp(arguments[index], "--", strlen("--")) == 0 ? folder : arguments[index];
    }
    if (argument_count > MOST_ARGUMENTS || (break_kind == BREAK_WORD && folder == NULL)) {
        (void)fprintf(stderr, "usage: model_v41_card_test [--break-word | --break-state] [<capture folder>]\n");
        return EXIT_FAILURE;
    }
    if (folder != NULL) {
        test_words(folder);
    } else {
        (void)printf("model_v41_card_test: no capture folder, the frequency words were not compared\n");
    }
    expect_status(backend->card_count(&card_count), STATUS_OK,
		  "the cards are counted");
    expect(card_count > 0 && card_count <= BACKEND_MAXIMUM_CARDS,
	   "the machine has cards for the test");
    expect_status(
	    backend->staging_allocate(STATE_BYTES + IMAGE_BYTES, &staging),
	    STATUS_OK, "page-locked memory for the read back and the image");
    for (uint32_t card_index = 0; staging != NULL && card_index < card_count; card_index++) {
        unsigned char *const image = (unsigned char *)staging + STATE_BYTES;
        for (uint64_t index = 0; index < IMAGE_BYTES; index++) {
            image[index] = (unsigned char)((index * IMAGE_SEED + card_index) >> IMAGE_SHIFT);
        }
        test_card(card_index, image, staging);
    }
    backend->staging_free(staging);
    if (break_kind != BREAK_NONE) {
        const bool shown = failures == 0 && reported == (break_kind == BREAK_WORD ? 1 : (int)card_count);
        (void)printf("model_v41_card_test: --break-%s %s\n", break_kind == BREAK_WORD ? "word" : "state",
                     shown ? "was reported, and nothing else failed" : "was NOT shown failing as it must be");
        return shown ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (failures != 0) {
        (void)fprintf(stderr, "model_v41_card_test FAILED: %d failures\n", failures);
        return EXIT_FAILURE;
    }
    (void)printf("model_v41_card_test passed on %" PRIu32 " cards\n", card_count);
    return EXIT_SUCCESS;
}
