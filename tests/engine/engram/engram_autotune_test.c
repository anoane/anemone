/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * The Engram stage of autotune (engram_autotune.c) and the profile's engram section (storage_profile.h):
 *   - the A D D A verdict: the lower geometric mean wins inside the noise, a level's loss beyond its own spread blocks
 *     the challenger and names the level, an exact tie goes to fewer cores and then to the default, the spread flag;
 *   - knob over profile over default, a parameter at a time; no section, or every table in RAM, changes nothing;
 *   - the stage's options: the other stages' refused by name, --tokens required, the ranges, before any file opens;
 *   - the section's read, write, replacement and qualification, every other member of the profile ("load" among
 *     them) kept byte for byte; a schema 2 profile or another source's refused; each stale and malformed reason;
 *   - the schedule (the seeded grid, the disturbed points again, the A D D A) and the state's resume on its identity;
 *   - the trace on the n-gram test pack of the model's geometry (TEST_MODEL_HEADS: two tables of
 *     DS41F_ENGRAM_ROWS_PER_TOKEN rows a token, as DeepSeek-V4.1-Flash's): its rows, the level shapes and the
 *     rotation's whole-block move;
 *   - a point of the stage's measurement over a store of two copies of that pack (under TMPDIR, else /var/tmp: the
 *     step I/O does not start on a tmpfs): every level, the identity check, and in checking builds a failed read
 *     (disturbed) and a changed byte (refused);
 *   - in the two-device scratch environment, the stage end to end beside a receipted replica.
 */
#include "engine/storage/autotune/storage_autotune.h"
#include "engine/cli/cli_autotune.h"
#include "engine/engram/engram_disk.h"
#include "common/hooks.h"
#include "engine/formats/pack/pack.h"
#include "engine/storage/replicas/replicas.h"
#include "engine/storage/profile/storage_profile.h"
#include "support/engram_pack.h"
#include "engine/engram/engram_step.h"
#include "support/log_expectation.h"
#include "models/deepseek_v41_flash/plan/model_v41.h"
#include <dirent.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <linux/io_uring.h>
#include <linux/magic.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/syscall.h>
#include <unistd.h>

#define PROFILE_BYTES (1u << 20u)
#define SECTION_BYTES 1024u
#define FIXTURE_KEY \
	"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
#define FIXTURE_DIGEST 0xaaull
#define FIXTURE_FILE_BYTES 8192ull
#define FIXTURE_SOURCE "/nonexistent/engram-section-source.anemone"
#define FIXTURE_STEPS 9u
#define FIXTURE_SLOTS 1024u
#define FIXTURE_THREADS 3u
#define FIXTURE_REPLACED_THREADS 1u
#define FIXTURE_SPIN_US 50u
#define FIXTURE_PIN 0u
#define KNOB_THREADS 4u
#define KNOB_SPIN_US 1000u
#define DISK_PERCENT 0u /* every table on disk */
#define PART_PERCENT 50u /* half the tables in RAM */
#define LOAD_MEMBER ",\"load\":{\"kept\":[1,2]}"
#define FILE_MODE (S_IRUSR | S_IWUSR)
#define STORE_KEYS 2u
#define OTHER_KEY \
	"fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
#define DIRECTORY_MODE (S_IRUSR | S_IWUSR | S_IXUSR)
#define CAPTURE_BYTES (64u << 10u)
#define LEVEL_NS 100000.0 /* a level's p50 in the verdict cases: 100 us */
#define LEVEL_NEAR 1000.0 /* 1 us: inside the spread */
#define LEVEL_HALF 500.0
#define LOSS_LEVEL 4u /* P1 */
#define LOSS_FACTOR 1.2 /* a 20 % loss */
#define SPREAD_FACTOR 1.3 /* one point 30 % off: over the 10 % bound */
#define FAST_FACTOR 0.5
#define SLOW_LEVEL_NS 20000000.0 /* 20 ms */
#define SLOW_LEVEL_LOW 19000000.0
#define SLOW_LEVEL_HIGH 21000000.0
#define SLOW_DEFAULT_LOW 18500000.0
#define SLOW_DEFAULT_HIGH 19500000.0
#define FEW_CORES 0.5
#define MORE_CORES 0.9
#define CASE_LONG_TOKENS 20u
#define CASE_SHORT_TOKENS 3u
#define TRACE_CASES 2u
#define TOKENS_ABOVE_MOST 17u /* one --tokens more than the stage takes */
#define OUT_OF_VOCABULARY 99u
#define ODD_BYTES 3u
#define TEST_ROW_COUNT \
	17u /* a table of two blocks, the second partly filled (the one-head test pack's table 1) */
#define TEST_ROW_MOVED 1u
#define WIDE_ROW_COUNT 15000u /* 1000 whole blocks */
#define WIDE_SHIFT 7919u
#define WIDE_ROW 7u
#define WIDE_MOVED 13792u /* block 919, row 7 of it */
#define STORE_COPIES 2u
#define RESUMED_POINTS 3u
#define DISTURBED_GRID_POINT 5u
#define CHEAP_NS 10u
#define DEAR_NS 1000u
#define DISTURBED_NS \
	1u /* a disturbed point's levels: cheaper than any quiet one */
#define ADDA_REMEASURED \
	2u /* the A D D A's third point, its second slot's again */
#define FIRST_DEFAULT_POINT \
	1u /* the A D D A's points: challenger, default, default, challenger */
#define SECOND_DEFAULT_POINT 2u
#define LAST_POINT (AUTOTUNE_ENGRAM_ABBA_POINTS - 1u)
#define DECODE_ROWS 8u /* the stage's decode step */
#define PREFILL_ROWS 1024u /* the stage's prefill chunk */
#define DECODE_LEVEL_TWO 1u /* D2: two decode streams */
#define DECODE_STREAMS_TWO 2u
#define CLI_FIXED_ARGUMENTS 4u /* anemone autotune PACK --stage=engram */
#define DIGEST_ROTATION 2u
#define STEP_FIELDS 2u /* a step: its first token, its rows */
#define COMPARE_ROTATION 3u
#define SEARCH_FOLDER "anemone_engram_rnd"
#define POINTS_FIRST_RUN 5u
#define STAGE_POINTS                   \
	(AUTOTUNE_ENGRAM_GRID_POINTS + \
	 AUTOTUNE_ENGRAM_ABBA_POINTS) /* a quiet run */
#define FIRST_TABLE_BLOCK \
	3u /* the test pack's tables: from block 3, each a run of whole blocks (engram_pack.c) */
#define TEST_MODEL_HEADS \
	DS41F_ENGRAM_ROWS_PER_TOKEN /* the test pack's look-back of 2: a head a row, the model's rows a token */
#define CORRUPT_OFFSET 100u
#define CAPTURE_LARGE (1u << 20u)
#define ENGRAM_STAGE_TURN \
	2u /* the run without --stage's lease check: the load stage's, then the Engram stage's */
#define ENGRAM_POINT_CUT \
	3u /* the Engram stage's lease check before its third point: two measured */

static unsigned failures;
static void expect(bool good, const char *message)
{
	if (!good) {
		failures++;
		(void)printf("FAILED: %s\n", message);
	}
}

static bool joined(char *output, size_t capacity, const char *first,
		   const char *second)
{
	const int length = snprintf(output, capacity, "%s%s", first, second);
	return length > 0 && (size_t)length < capacity;
}

/* Whether the folder at path holds no entry. */
static bool folder_empty(const char *path)
{
	DIR *const folder = opendir(path);
	bool empty = folder != NULL;
	for (const struct dirent *entry = empty ? readdir(folder) : NULL;
	     entry != NULL; entry = readdir(folder))
		empty = empty && (strcmp(entry->d_name, ".") == 0 ||
				  strcmp(entry->d_name, "..") == 0);
	if (folder != NULL)
		(void)closedir(folder);
	return empty;
}

static bool file_write(const char *path, const void *bytes, size_t count)
{
	const int descriptor =
		open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, FILE_MODE);
	const bool written = descriptor >= 0 &&
			     write(descriptor, bytes, count) ==
				     (ssize_t)count &&
			     fsync(descriptor) == 0;
	if (descriptor >= 0)
		(void)close(descriptor);
	return written;
}

static bool file_text(const char *path, char *text, size_t capacity)
{
	FILE *const input = fopen(path, "re");
	const size_t bytes =
		input != NULL ? fread(text, 1, capacity - 1u, input) : 0;
	text[bytes] = '\0';
	if (input != NULL)
		(void)fclose(input);
	return input != NULL;
}

static uint32_t occurrences(const char *text, const char *searched)
{
	uint32_t count = 0;
	for (const char *found = strstr(text, searched); found != NULL;
	     found = strstr(found + strlen(searched), searched))
		count++;
	return count;
}

/* Whether two choices hold the same values and sources. */
static bool choice_same(const struct storage_engram_choice *first,
			const struct storage_engram_choice *second)
{
	return first->threads_per_copy == second->threads_per_copy &&
	       first->spin_us == second->spin_us && first->pin == second->pin &&
	       first->threads_from_profile == second->threads_from_profile &&
	       first->spin_from_profile == second->spin_from_profile &&
	       first->pin_from_profile == second->pin_from_profile;
}

static void choice_cases(void)
{
	const struct storage_engram_section section = {
		.threads_per_copy = FIXTURE_THREADS,
		.spin_us = FIXTURE_SPIN_US,
		.pin = FIXTURE_PIN
	};
	const struct storage_engram_choice defaults = {
		.threads_per_copy = ENGRAM_STEP_IO_THREADS_DEFAULT,
		.spin_us = ENGRAM_STEP_SPIN_US_DEFAULT,
		.pin = 1
	};
	struct storage_engram_choice choice = defaults;
	storage_engram_choose(NULL, DISK_PERCENT, false, false, false, &choice);
	expect(choice_same(&choice, &defaults),
	       "no applicable section leaves every value as the knob or the default gave it");
	choice = defaults;
	storage_engram_choose(&section, ENGRAM_RAM_PERCENT, false, false, false,
			      &choice);
	expect(choice_same(&choice, &defaults),
	       "with every table in RAM the section is not applied");
	choice = defaults;
	storage_engram_choose(&section, DISK_PERCENT, false, false, false,
			      &choice);
	expect(choice.threads_per_copy == FIXTURE_THREADS &&
		       choice.spin_us == FIXTURE_SPIN_US &&
		       choice.pin == FIXTURE_PIN &&
		       choice.threads_from_profile &&
		       choice.spin_from_profile && choice.pin_from_profile,
	       "an applicable section gives every parameter no knob gave, from disk tables");
	const struct storage_engram_choice knobs = { .threads_per_copy =
							     KNOB_THREADS,
						     .spin_us = KNOB_SPIN_US,
						     .pin = 1 };
	choice = knobs;
	storage_engram_choose(&section, DISK_PERCENT, true, false, false,
			      &choice);
	expect(choice.threads_per_copy == KNOB_THREADS &&
		       !choice.threads_from_profile &&
		       choice.spin_us == FIXTURE_SPIN_US &&
		       choice.pin == FIXTURE_PIN,
	       "a threads knob wins over the section, which still gives the spin and the pin");
	choice = knobs;
	storage_engram_choose(&section, DISK_PERCENT, false, true, false,
			      &choice);
	expect(choice.spin_us == KNOB_SPIN_US && !choice.spin_from_profile &&
		       choice.threads_per_copy == FIXTURE_THREADS &&
		       choice.pin == FIXTURE_PIN,
	       "a spin knob wins over the section, which still gives the threads and the pin");
	choice = knobs;
	storage_engram_choose(&section, DISK_PERCENT, false, false, true,
			      &choice);
	expect(choice.pin == 1u && !choice.pin_from_profile &&
		       choice.threads_per_copy == FIXTURE_THREADS &&
		       choice.spin_us == FIXTURE_SPIN_US,
	       "a pin knob wins over the section, which still gives the threads and the spin");
}

static bool section_text(char *output, size_t capacity, uint32_t threads,
			 const char *key, const char *interface,
			 uint32_t spin_us)
{
	const int length = snprintf(
		output, capacity,
		"{\"schema\":%u,\"enabled\":0,\"qualification\":\"candidate\",\"created_unix\":1,"
		"\"pack\":{\"header_digest\":\"%016" PRIx64
		"\",\"file_byte_count\":%" PRIu64 "},"
		"\"copies\":[{\"key\":\"%s\",\"path\":\"/x\",\"filesystem\":\"xfs\"}],"
		"\"shape\":{\"interface\":\"%s\",\"steps\":%u,\"slots\":%u},"
		"\"step_io\":{\"threads_per_copy\":%u,\"spin_us\":%u,\"pin\":%u}}",
		STORAGE_ENGRAM_SECTION_SCHEMA, (uint64_t)FIXTURE_DIGEST,
		(uint64_t)FIXTURE_FILE_BYTES, key, interface, FIXTURE_STEPS,
		FIXTURE_SLOTS, threads, spin_us, FIXTURE_PIN);
	return length > 0 && (size_t)length < capacity;
}

static bool section_written(const char *profile, uint32_t threads,
			    const char *key, const char *interface,
			    uint32_t spin_us, const char **refusal)
{
	char section[SECTION_BYTES];
	return section_text(section, sizeof(section), threads, key, interface,
			    spin_us) &&
	       storage_stage_section_write(profile, FIXTURE_SOURCE,
					   STORAGE_ENGRAM_SECTION_NAME, section,
					   strlen(section),
					   refusal) == STATUS_OK;
}

/* A malformed or stale section: its reason. */
static void reason_cases(const char *profile)
{
	char reason[STORAGE_PROFILE_REASON_BYTES];
	struct storage_engram_section section;
	const char *refusal = NULL;
	static const struct {
		uint32_t threads, spin_us;
		const char *key, *interface, *what;
	} malformed[] = {
		{ 0, FIXTURE_SPIN_US, FIXTURE_KEY, STORAGE_ENGRAM_INTERFACE,
		  "no thread a copy is malformed" },
		{ STORAGE_FLIGHT_THREADS_PER_COPY_MOST + 1u, FIXTURE_SPIN_US,
		  FIXTURE_KEY, STORAGE_ENGRAM_INTERFACE,
		  "more threads a copy than the flight takes is malformed" },
		{ FIXTURE_THREADS, ENGRAM_STEP_SPIN_US_MOST + 1u, FIXTURE_KEY,
		  STORAGE_ENGRAM_INTERFACE,
		  "a spin above the knob's range is malformed" },
		{ FIXTURE_THREADS, FIXTURE_SPIN_US, "not-a-key",
		  STORAGE_ENGRAM_INTERFACE,
		  "a copy key that is no digest is malformed" },
		{ FIXTURE_THREADS, FIXTURE_SPIN_US, FIXTURE_KEY, "uring_poll",
		  "another interface is malformed" }
	};
	for (size_t index = 0; index < sizeof(malformed) / sizeof(malformed[0]);
	     index++) {
		(void)unlink(profile);
		expect(section_written(profile, malformed[index].threads,
				       malformed[index].key,
				       malformed[index].interface,
				       malformed[index].spin_us, &refusal) &&
			       !storage_engram_section_read(profile, &section,
							    reason) &&
			       strcmp(reason, "engram_section_malformed") == 0,
		       malformed[index].what);
	}
	(void)unlink(profile);
	static const char no_section[] =
		"{\"schema\":3,\"source\":\"" FIXTURE_SOURCE "\"" LOAD_MEMBER
		"}\n";
	expect(file_write(profile, no_section, sizeof(no_section) - 1u) &&
		       !storage_engram_section_read(profile, &section,
						    reason) &&
		       strcmp(reason, "no_engram_section") == 0,
	       "a stage profile without the member: no_engram_section");
	(void)unlink(profile);
	expect(section_written(profile, FIXTURE_THREADS, FIXTURE_KEY,
			       STORAGE_ENGRAM_INTERFACE, FIXTURE_SPIN_US,
			       &refusal) &&
		       storage_engram_section_read(profile, &section, reason),
	       "the stale cases' section reads");
	const struct pack_header stale = { .header_digest = FIXTURE_DIGEST + 1u,
					   .file_byte_count =
						   FIXTURE_FILE_BYTES };
	const struct pack_header same = { .header_digest = FIXTURE_DIGEST,
					  .file_byte_count =
						  FIXTURE_FILE_BYTES };
	expect(!storage_engram_section_matches(&section, &stale, NULL, 0,
					       FIXTURE_STEPS, FIXTURE_SLOTS,
					       reason) &&
		       strcmp(reason, "engram_section_stale_pack") == 0,
	       "a section of another pack is stale");
	expect(!storage_engram_section_matches(&section, &same, NULL, 0,
					       FIXTURE_STEPS, FIXTURE_SLOTS,
					       reason) &&
		       strcmp(reason, "engram_section_stale_copies") == 0,
	       "a section of other copies is stale");
	/* the shape after the copies (no copy here, so no drive needs a key) */
	struct storage_engram_section shaped = section;
	shaped.copy_count = 0;
	expect(storage_engram_section_matches(&shaped, &same, NULL, 0,
					      FIXTURE_STEPS, FIXTURE_SLOTS,
					      reason) &&
		       !storage_engram_section_matches(&shaped, &same, NULL, 0,
						       FIXTURE_STEPS + 1u,
						       FIXTURE_SLOTS, reason) &&
		       strcmp(reason, "engram_section_stale_shape") == 0 &&
		       !storage_engram_section_matches(
			       &shaped, &same, NULL, 0, FIXTURE_STEPS,
			       FIXTURE_SLOTS + 1u, reason) &&
		       strcmp(reason, "engram_section_stale_shape") == 0,
	       "a section measured in another flight shape is stale");
	/* two copies' keys in their order, as the stage measured them */
	struct storage_engram_section two = section;
	two.copy_count = STORE_KEYS;
	(void)snprintf(two.keys[1], sizeof(two.keys[1]), "%s", OTHER_KEY);
	const char ordered[STORE_KEYS][STORAGE_PROFILE_KEY_BYTES] = {
		FIXTURE_KEY, OTHER_KEY
	};
	const char swapped[STORE_KEYS][STORAGE_PROFILE_KEY_BYTES] = {
		OTHER_KEY, FIXTURE_KEY
	};
	const char unread[STORE_KEYS][STORAGE_PROFILE_KEY_BYTES] = {
		FIXTURE_KEY, ""
	};
	expect(storage_engram_section_matches(&two, &same, ordered, STORE_KEYS,
					      FIXTURE_STEPS, FIXTURE_SLOTS,
					      reason) &&
		       !storage_engram_section_matches(
			       &two, &same, swapped, STORE_KEYS, FIXTURE_STEPS,
			       FIXTURE_SLOTS, reason) &&
		       strcmp(reason, "engram_section_stale_copies") == 0 &&
		       !storage_engram_section_matches(&two, &same, ordered, 1,
						       FIXTURE_STEPS,
						       FIXTURE_SLOTS, reason) &&
		       strcmp(reason, "engram_section_stale_copies") == 0 &&
		       !storage_engram_section_matches(
			       &two, &same, unread, STORE_KEYS, FIXTURE_STEPS,
			       FIXTURE_SLOTS, reason) &&
		       strcmp(reason, "engram_section_stale_copies") == 0,
	       "a section matches its copies' keys in their order only: swapped, fewer or an unread key are stale");
	(void)unlink(profile);
}

/* Serving's application of the section (storage_engram_section_apply, engram_steps_start's seam): a candidate never,
 * a stale section never, by its reason, knob over profile over default a parameter at a time, and with every table in
 * RAM no read at all; the RAM share is serving's own. */
static void serving_cases(const char *folder)
{
	char profile[PATH_MAX], absent[PATH_MAX],
		reason[STORAGE_PROFILE_REASON_BYTES];
	const char *refusal = NULL;
	const struct pack_header same = { .header_digest = FIXTURE_DIGEST,
					  .file_byte_count =
						  FIXTURE_FILE_BYTES };
	const struct pack_header stale = { .header_digest = FIXTURE_DIGEST + 1u,
					   .file_byte_count =
						   FIXTURE_FILE_BYTES };
	const char keys[1][STORAGE_PROFILE_KEY_BYTES] = { FIXTURE_KEY };
	const char other[1][STORAGE_PROFILE_KEY_BYTES] = { OTHER_KEY };
	const struct storage_engram_choice defaults = {
		.threads_per_copy = ENGRAM_STEP_IO_THREADS_DEFAULT,
		.spin_us = ENGRAM_STEP_SPIN_US_DEFAULT,
		.pin = 1
	};
	const struct storage_engram_choice knobs = { .threads_per_copy =
							     KNOB_THREADS,
						     .spin_us = KNOB_SPIN_US,
						     .pin = 1 };
	const struct storage_engram_serving base = { .profile = profile,
						     .header = &same,
						     .copy_keys = keys,
						     .copy_count = 1,
						     .steps = FIXTURE_STEPS,
						     .slots = FIXTURE_SLOTS,
						     .ram_percent =
							     DISK_PERCENT };
	struct storage_engram_serving serving = base;
	struct storage_engram_choice choice = defaults;
	if (!joined(profile, sizeof(profile), folder, "/serving.json") ||
	    !joined(absent, sizeof(absent), folder, "/absent.json")) {
		expect(false, "the serving cases' paths fit");
		return;
	}
	expect(section_written(profile, FIXTURE_THREADS, FIXTURE_KEY,
			       STORAGE_ENGRAM_INTERFACE, FIXTURE_SPIN_US,
			       &refusal),
	       "the serving cases' candidate section is written");
	storage_engram_section_apply(&serving, &choice, reason);
	expect(strcmp(reason, "engram_section_candidate") == 0 &&
		       choice_same(&choice, &defaults),
	       "serving never applies a candidate section (enabled 0): the defaults hold, by name");
	expect(storage_stage_section_qualify(profile, FIXTURE_SOURCE,
					     STORAGE_ENGRAM_SECTION_NAME,
					     &refusal) == STATUS_OK,
	       "the serving cases' section is qualified");
	storage_engram_section_apply(&serving, &choice, reason);
	expect(reason[0] == '\0' &&
		       choice.threads_per_copy == FIXTURE_THREADS &&
		       choice.spin_us == FIXTURE_SPIN_US &&
		       choice.pin == FIXTURE_PIN &&
		       choice.threads_from_profile &&
		       choice.spin_from_profile && choice.pin_from_profile,
	       "serving applies an enabled section of this pack, these copies and this shape, from disk tables");
	enum serving_change {
		SERVING_PACK,
		SERVING_KEY,
		SERVING_FEWER,
		SERVING_STEPS,
		SERVING_SLOTS
	};
	static const struct {
		enum serving_change change;
		const char *expected, *what;
	} stale_reasons[] = {
		{ SERVING_PACK, "engram_section_stale_pack",
		  "serving never applies a section of another pack" },
		{ SERVING_KEY, "engram_section_stale_copies",
		  "serving never applies a section of another copy's drive" },
		{ SERVING_FEWER, "engram_section_stale_copies",
		  "serving never applies a section of more copies than the store admits" },
		{ SERVING_STEPS, "engram_section_stale_shape",
		  "serving never applies a section of another flight's steps" },
		{ SERVING_SLOTS, "engram_section_stale_shape",
		  "serving never applies a section of another flight's slots" }
	};
	for (uint32_t index = 0;
	     index < sizeof(stale_reasons) / sizeof(stale_reasons[0]);
	     index++) {
		const enum serving_change change = stale_reasons[index].change;
		serving = base;
		serving.header = change == SERVING_PACK ? &stale : &same;
		serving.copy_keys = change == SERVING_KEY ? other : keys;
		serving.copy_count = change == SERVING_FEWER ? 0 : 1;
		serving.steps =
			FIXTURE_STEPS + (change == SERVING_STEPS ? 1u : 0u);
		serving.slots =
			FIXTURE_SLOTS + (change == SERVING_SLOTS ? 1u : 0u);
		choice = defaults;
		storage_engram_section_apply(&serving, &choice, reason);
		expect(strcmp(reason, stale_reasons[index].expected) == 0 &&
			       choice_same(&choice, &defaults),
		       stale_reasons[index].what);
	}
	serving = base;
	serving.explicit_threads = true;
	choice = knobs;
	storage_engram_section_apply(&serving, &choice, reason);
	const bool threads_knob = choice.threads_per_copy == KNOB_THREADS &&
				  !choice.threads_from_profile &&
				  choice.spin_us == FIXTURE_SPIN_US &&
				  choice.pin == FIXTURE_PIN;
	serving = base;
	serving.explicit_spin = true;
	choice = knobs;
	storage_engram_section_apply(&serving, &choice, reason);
	const bool spin_knob = choice.spin_us == KNOB_SPIN_US &&
			       !choice.spin_from_profile &&
			       choice.threads_per_copy == FIXTURE_THREADS &&
			       choice.pin == FIXTURE_PIN;
	serving = base;
	serving.explicit_pin = true;
	choice = knobs;
	storage_engram_section_apply(&serving, &choice, reason);
	expect(threads_knob && spin_knob && choice.pin == 1u &&
		       !choice.pin_from_profile &&
		       choice.threads_per_copy == FIXTURE_THREADS &&
		       choice.spin_us == FIXTURE_SPIN_US,
	       "at serving each knob given wins over the section, which gives the parameters no knob gave");
	serving = base;
	serving.ram_percent = PART_PERCENT;
	choice = defaults;
	storage_engram_section_apply(&serving, &choice, reason);
	expect(reason[0] == '\0' && choice.threads_from_profile &&
		       serving.ram_percent == PART_PERCENT,
	       "a section measured with no RAM share applies beside serving's RAM share, which stays the RAM fit's");
	serving = base;
	serving.ram_percent = ENGRAM_RAM_PERCENT;
	choice = defaults;
	storage_engram_section_apply(&serving, &choice, reason);
	const bool kept = strcmp(reason, "tables_in_ram") == 0 &&
			  choice_same(&choice, &defaults);
	serving.profile = absent;
	storage_engram_section_apply(&serving, &choice, reason);
	expect(kept && strcmp(reason, "tables_in_ram") == 0 &&
		       choice_same(&choice, &defaults),
	       "with every table in RAM serving reads no section: an enabled one is not applied, a missing one not "
	       "looked for");
	serving = base;
	serving.profile = NULL;
	storage_engram_section_apply(&serving, &choice, reason);
	const bool none = strcmp(reason, "profile_off") == 0;
	serving.profile = "";
	storage_engram_section_apply(&serving, &choice, reason);
	expect(none && strcmp(reason, "profile_off") == 0 &&
		       choice_same(&choice, &defaults),
	       "without a profile serving keeps the defaults: profile_off");
	(void)unlink(profile);
}

static void section_cases(const char *folder)
{
	char profile[PATH_MAX], text[PROFILE_BYTES], before[PROFILE_BYTES];
	char reason[STORAGE_PROFILE_REASON_BYTES];
	struct storage_engram_section section;
	const char *refusal = NULL;
	if (!joined(profile, sizeof(profile), folder,
		    "/storage-profile.json")) {
		expect(false, "the section cases' profile path fits");
		return;
	}
	expect(!storage_engram_section_read(profile, &section, reason) &&
		       strcmp(reason, "missing") == 0,
	       "no profile: no section, reason missing");
	expect(section_written(profile, FIXTURE_THREADS, FIXTURE_KEY,
			       STORAGE_ENGRAM_INTERFACE, FIXTURE_SPIN_US,
			       &refusal) &&
		       storage_engram_section_read(profile, &section, reason) &&
		       section.enabled == 0 && section.copy_count == 1u &&
		       section.threads_per_copy == FIXTURE_THREADS &&
		       section.spin_us == FIXTURE_SPIN_US &&
		       section.pin == FIXTURE_PIN &&
		       section.steps == FIXTURE_STEPS &&
		       section.slots == FIXTURE_SLOTS &&
		       section.file_byte_count == FIXTURE_FILE_BYTES &&
		       strcmp(section.keys[0], FIXTURE_KEY) == 0 &&
		       strcmp(section.source, FIXTURE_SOURCE) == 0,
	       "a new profile is a schema 3 profile of the source with the engram section as written");
	static const char kept[] = "{\"schema\":3,\"source\":\"" FIXTURE_SOURCE
				   "\"" LOAD_MEMBER "}\n";
	expect(file_write(profile, kept, sizeof(kept) - 1u) &&
		       section_written(profile, FIXTURE_THREADS, FIXTURE_KEY,
				       STORAGE_ENGRAM_INTERFACE,
				       FIXTURE_SPIN_US, &refusal) &&
		       file_text(profile, text, sizeof(text)) &&
		       strstr(text, LOAD_MEMBER ",\"engram\":{") != NULL,
	       "the engram section is added after the load member, which stays byte for byte");
	expect(section_written(profile, FIXTURE_REPLACED_THREADS, FIXTURE_KEY,
			       STORAGE_ENGRAM_INTERFACE, FIXTURE_SPIN_US,
			       &refusal) &&
		       file_text(profile, text, sizeof(text)) &&
		       strstr(text, LOAD_MEMBER ",\"engram\":{") != NULL &&
		       occurrences(text, "\"engram\":") == 1u &&
		       storage_engram_section_read(profile, &section, reason) &&
		       section.threads_per_copy == FIXTURE_REPLACED_THREADS,
	       "a section written again replaces the old one in place, the load member kept");
	expect(storage_stage_section_qualify(profile, FIXTURE_SOURCE,
					     STORAGE_LOAD_SECTION_NAME,
					     &refusal) != STATUS_OK &&
		       refusal != NULL &&
		       strstr(refusal, "no candidate load section") != NULL,
	       "the load stage's qualify finds no candidate load section in another stage's member");
	expect(storage_stage_section_qualify(profile, FIXTURE_SOURCE,
					     STORAGE_ENGRAM_SECTION_NAME,
					     &refusal) == STATUS_OK &&
		       storage_engram_section_read(profile, &section, reason) &&
		       section.enabled == 1u &&
		       file_text(profile, text, sizeof(text)) &&
		       strstr(text,
			      "\"enabled\":1,\"qualification\":\"qualified\"") !=
			       NULL &&
		       strstr(text, LOAD_MEMBER) != NULL,
	       "qualifying enables the engram candidate and changes nothing else");
	expect(storage_stage_section_qualify(profile, FIXTURE_SOURCE,
					     STORAGE_ENGRAM_SECTION_NAME,
					     &refusal) != STATUS_OK &&
		       refusal != NULL &&
		       strstr(refusal, "no candidate engram section") != NULL,
	       "a qualified section is no candidate to qualify again");
	expect(storage_stage_section_write(profile, FIXTURE_SOURCE, "tune",
					   "{}", strlen("{}"), &refusal) ==
			       STATUS_INVALID_ARGUMENT &&
		       storage_stage_section_qualify(profile, FIXTURE_SOURCE,
						     "tune", &refusal) ==
			       STATUS_INVALID_ARGUMENT,
	       "a member of no stage is refused");
	static const char combined[] =
		"{\"schema\":2,\"source\":\"" FIXTURE_SOURCE "\"}\n";
	expect(file_write(profile, combined, sizeof(combined) - 1u) &&
		       storage_stage_section_writable(profile, FIXTURE_SOURCE,
						      &refusal) != STATUS_OK &&
		       refusal != NULL &&
		       strstr(refusal, "no stage profile") != NULL &&
		       !section_written(profile, FIXTURE_THREADS, FIXTURE_KEY,
					STORAGE_ENGRAM_INTERFACE,
					FIXTURE_SPIN_US, &refusal) &&
		       file_text(profile, before, sizeof(before)) &&
		       strcmp(before, combined) == 0 &&
		       !storage_engram_section_read(profile, &section,
						    reason) &&
		       strcmp(reason, "not_a_stage_profile") == 0,
	       "a schema 2 profile is left as it is, and holds no engram section for serving");
	static const char other[] =
		"{\"schema\":3,\"source\":\"/nonexistent/other.anemone\"}\n";
	expect(file_write(profile, other, sizeof(other) - 1u) &&
		       !section_written(profile, FIXTURE_THREADS, FIXTURE_KEY,
					STORAGE_ENGRAM_INTERFACE,
					FIXTURE_SPIN_US, &refusal) &&
		       refusal != NULL &&
		       strstr(refusal, "another source") != NULL,
	       "another source's profile is refused");
	reason_cases(profile);
}

/* The A D D A's first-table p50s, point-major (autotune_engram_verdict): every level of a point at its value. */
#define AT(point, level) ((point) * AUTOTUNE_ENGRAM_LEVELS + (level))
static void points_fill(
	double first_ns[AUTOTUNE_ENGRAM_ABBA_POINTS * AUTOTUNE_ENGRAM_LEVELS],
	const double values[AUTOTUNE_ENGRAM_ABBA_POINTS])
{
	for (uint32_t point = 0; point < AUTOTUNE_ENGRAM_ABBA_POINTS; point++)
		for (uint32_t level = 0; level < AUTOTUNE_ENGRAM_LEVELS;
		     level++)
			first_ns[AT(point, level)] = values[point];
}

static void verdict_cases(void)
{
	double first_ns[AUTOTUNE_ENGRAM_ABBA_POINTS * AUTOTUNE_ENGRAM_LEVELS];
	const double same_cores[AUTOTUNE_ENGRAM_ABBA_POINTS] = {
		MORE_CORES, MORE_CORES, MORE_CORES, MORE_CORES
	};
	struct autotune_engram_verdict verdict;
	/* the order: challenger, default, default, challenger */
	const double within[AUTOTUNE_ENGRAM_ABBA_POINTS] = {
		LEVEL_NS - LEVEL_NEAR, LEVEL_NS + LEVEL_HALF,
		LEVEL_NS + LEVEL_NEAR + LEVEL_HALF, LEVEL_NS + LEVEL_NEAR
	};
	points_fill(first_ns, within);
	autotune_engram_verdict(first_ns, same_cores, &verdict);
	expect(!verdict.default_holds && !verdict.exact_tie &&
		       verdict.blocked_by == AUTOTUNE_ENGRAM_LEVELS &&
		       verdict.margin > 0.0 &&
		       verdict.margin < verdict.level_spread[0] &&
		       !verdict.spread_over_bound,
	       "the lower geometric mean wins inside the noise (the owner's rule), its margin and spread recorded");
	const double faster[AUTOTUNE_ENGRAM_ABBA_POINTS] = {
		LEVEL_NS * FAST_FACTOR, LEVEL_NS, LEVEL_NS,
		LEVEL_NS * FAST_FACTOR
	};
	points_fill(first_ns, faster);
	first_ns[AT(0, LOSS_LEVEL)] = first_ns[AT(LAST_POINT, LOSS_LEVEL)] =
		LEVEL_NS * LOSS_FACTOR;
	autotune_engram_verdict(first_ns, same_cores, &verdict);
	expect(verdict.challenger_cost < verdict.default_cost &&
		       verdict.default_holds &&
		       verdict.blocked_by == LOSS_LEVEL &&
		       verdict.level_loss[LOSS_LEVEL] >
			       verdict.level_spread[LOSS_LEVEL],
	       "a level slower than the default beyond its own spread blocks a cheaper challenger and is named");
	const double tie[AUTOTUNE_ENGRAM_ABBA_POINTS] = { LEVEL_NS, LEVEL_NS,
							  LEVEL_NS, LEVEL_NS };
	points_fill(first_ns, tie);
	autotune_engram_verdict(first_ns, same_cores, &verdict);
	expect(verdict.exact_tie && verdict.default_holds,
	       "an exact tie with equal cores goes to the default");
	const double challenger_lighter[AUTOTUNE_ENGRAM_ABBA_POINTS] = {
		FEW_CORES, MORE_CORES, MORE_CORES, FEW_CORES
	};
	autotune_engram_verdict(first_ns, challenger_lighter, &verdict);
	expect(verdict.exact_tie && !verdict.default_holds,
	       "an exact tie goes to the arm of fewer cores");
	const double spread[AUTOTUNE_ENGRAM_ABBA_POINTS] = {
		LEVEL_NS * FAST_FACTOR, LEVEL_NS, LEVEL_NS * SPREAD_FACTOR,
		LEVEL_NS * FAST_FACTOR
	};
	points_fill(first_ns, spread);
	autotune_engram_verdict(first_ns, same_cores, &verdict);
	expect(verdict.spread_over_bound && !verdict.default_holds &&
		       verdict.spread > AUTOTUNE_ENGRAM_SPREAD_BOUND,
	       "a spread over the bound is flagged, and the verdict stands");
	/* level 0 twice as fast, the slow level 5 % slower inside its 10 % spread: the arithmetic mean of the levels
	 * (dominated by the 20 ms level) would keep the default; the geometric mean takes the challenger */
	points_fill(first_ns, tie);
	first_ns[AT(0, 0)] = first_ns[AT(LAST_POINT, 0)] =
		LEVEL_NS * FAST_FACTOR;
	first_ns[AT(0, LOSS_LEVEL)] = SLOW_LEVEL_LOW;
	first_ns[AT(LAST_POINT, LOSS_LEVEL)] = SLOW_LEVEL_HIGH;
	first_ns[AT(FIRST_DEFAULT_POINT, LOSS_LEVEL)] = SLOW_DEFAULT_LOW;
	first_ns[AT(SECOND_DEFAULT_POINT, LOSS_LEVEL)] = SLOW_DEFAULT_HIGH;
	double challenger_sum = 0.0, default_sum = 0.0;
	for (uint32_t level = 0; level < AUTOTUNE_ENGRAM_LEVELS; level++) {
		challenger_sum += first_ns[AT(0, level)] +
				  first_ns[AT(LAST_POINT, level)];
		default_sum += first_ns[AT(FIRST_DEFAULT_POINT, level)] +
			       first_ns[AT(SECOND_DEFAULT_POINT, level)];
	}
	autotune_engram_verdict(first_ns, same_cores, &verdict);
	expect(challenger_sum > default_sum && !verdict.default_holds &&
		       verdict.blocked_by == AUTOTUNE_ENGRAM_LEVELS &&
		       SLOW_LEVEL_NS > SLOW_DEFAULT_HIGH,
	       "where the geometric and arithmetic means disagree, the geometric mean decides");
}

/* anemone autotune's command (--stage=engram) with its stderr captured whole into text through capture_path: its exit
 * status, or -1. */
static int engram_captured(int count, char **arguments,
			   const char *capture_path, char *text,
			   size_t capacity)
{
	text[0] = '\0';
	(void)fflush(stderr);
	const int capture = open(
		capture_path, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, FILE_MODE);
	const int saved = dup(STDERR_FILENO);
	int status = -1;
	if (capture >= 0 && saved >= 0 &&
	    dup2(capture, STDERR_FILENO) == STDERR_FILENO) {
		status = storage_autotune_cli(count, arguments,
					      cli_autotune_stages,
					      CLI_AUTOTUNE_STAGES);
		(void)fflush(stderr);
		if (dup2(saved, STDERR_FILENO) != STDERR_FILENO ||
		    test_log_capture_read(capture, text, capacity) < 0)
			status = -1;
	}
	if (saved >= 0)
		(void)close(saved);
	if (capture >= 0) {
		(void)close(capture);
		(void)unlink(capture_path);
	}
	return status;
}

/* Each refused before any file opens (the pack does not exist), its reason and the option named. */
static void option_cases(const char *folder)
{
	char capture[PATH_MAX], text[CAPTURE_BYTES];
	if (!joined(capture, sizeof(capture), folder, "/capture.txt")) {
		expect(false, "the option cases' capture path fits");
		return;
	}
#define FOREIGN \
	"autotune: refused: the copy stage's or the load stage's option, not the Engram stage's: "
	static const struct {
		const char *option, *expected;
	} refused[] = {
		{ "--target=/nonexistent/replica.anemone",
		  FOREIGN "--target=" },
		{ "--copy_workers=2", FOREIGN "--copy_workers=2" },
		{ "--copy_hash_pairs=1", FOREIGN "--copy_hash_pairs=1" },
		{ "--manifest=/nonexistent/m.json", FOREIGN "--manifest=" },
		{ "--verify=full", FOREIGN "--verify=full" },
		{ "--replica_floor_percent=20",
		  FOREIGN "--replica_floor_percent=20" },
		{ "--window_gib=1", FOREIGN "--window_gib=1" },
		{ "--prompt=1", "autotune: refused: " AUTOTUNE_PROMPT_RETIRED
				": --prompt=1" },
		{ "--seconds=0",
		  "autotune: refused: --seconds takes 1 to 1650: --seconds=0" },
		{ "--seconds=1651",
		  "autotune: refused: --seconds takes 1 to 1650: --seconds=1651" },
		{ "--replica_external=2",
		  "autotune: refused: --replica_external takes 0 or 1: --replica_external=2" },
		{ "--frobnicate",
		  "autotune: refused: not an option of the Engram stage: --frobnicate" }
	};
	char program[] = "anemone", command[] = "autotune",
	     source[] = "/nonexistent/engram-stage.anemone",
	     stage[] = "--stage=engram",
	     tokens[] = "--tokens=/nonexistent/tokens.u32", option[PATH_MAX];
	for (size_t index = 0; index < sizeof(refused) / sizeof(refused[0]);
	     index++) {
		(void)snprintf(option, sizeof(option), "%s",
			       refused[index].option);
		char *arguments[] = { program, command, source,
				      stage,   tokens,	option };
		const int status = engram_captured(
			(int)(sizeof(arguments) / sizeof(arguments[0])),
			arguments, capture, text, sizeof(text));
		expect(status == EXIT_FAILURE &&
			       strstr(text, refused[index].expected) != NULL,
		       refused[index].expected);
	}
	char *no_tokens[] = { program, command, source, stage };
	expect(engram_captured((int)(sizeof(no_tokens) / sizeof(no_tokens[0])),
			       no_tokens, capture, text,
			       sizeof(text)) == EXIT_FAILURE &&
		       strstr(text,
			      "autotune: refused: the Engram stage replays real prompts: give "
			      "--tokens=TOKEN_IDS_U32: --tokens") != NULL,
	       "without --tokens the stage is refused by name: it replays real prompts");
	char *many[TOKENS_ABOVE_MOST + CLI_FIXED_ARGUMENTS] = { program,
								command, source,
								stage };
	for (uint32_t index = 0; index < TOKENS_ABOVE_MOST; index++)
		many[CLI_FIXED_ARGUMENTS + index] = tokens;
	expect(engram_captured((int)(sizeof(many) / sizeof(many[0])), many,
			       capture, text, sizeof(text)) == EXIT_FAILURE &&
		       strstr(text,
			      "autotune: refused: at most 16 --tokens files") !=
			       NULL,
	       "a seventeenth --tokens file is refused");
	char help[] = "--help";
	char *helped[] = { program, command, source, stage, help };
	expect(engram_captured((int)(sizeof(helped) / sizeof(helped[0])),
			       helped, capture, text,
			       sizeof(text)) == EXIT_SUCCESS &&
		       strstr(text,
			      "usage: anemone autotune PACK --stage=engram") !=
			       NULL &&
		       strstr(text, "refused") == NULL,
	       "--stage=engram --help prints the usage alone and exits 0");
}

/* A state of the test identity with count points where the schedule puts them, every one quiet. */
static void state_build(struct autotune_engram_state *state, uint32_t count)
{
	memset(state, 0, sizeof(*state));
	state->schema = 1;
	state->copy_count = 1;
	state->steps = FIXTURE_STEPS;
	state->slots = FIXTURE_SLOTS;
	state->file_byte_count = FIXTURE_FILE_BYTES;
	state->seed = FIXTURE_DIGEST;
	(void)snprintf(state->pack_identity, sizeof(state->pack_identity),
		       "%016" PRIx64, (uint64_t)FIXTURE_DIGEST);
	(void)snprintf(state->trace_identity, sizeof(state->trace_identity),
		       "%s", FIXTURE_KEY);
	(void)snprintf(state->keys[0], sizeof(state->keys[0]), "%s",
		       FIXTURE_KEY);
	for (uint32_t index = 0; index < count; index++) {
		uint32_t arm = 0, phase = 0;
		const char *refusal = NULL;
		if (!autotune_engram_next(state, &arm, &phase, &refusal))
			return;
		struct autotune_engram_point *const point =
			&state->points[state->point_count++];
		point->arm_index = arm;
		point->phase = phase;
		point->rotation = index;
		autotune_engram_arm_of(arm, &point->arm);
		for (uint32_t level = 0; level < AUTOTUNE_ENGRAM_LEVELS;
		     level++)
			point->levels[level].first_p50_ns =
				arm == RESUMED_POINTS ? CHEAP_NS : DEAR_NS;
	}
}

/* The next point the schedule gives, appended with disturbance: its levels CHEAP_NS for the cheapest grid arm, else
 * DEAR_NS (a disturbed point's levels DISTURBED_NS, which no verdict may see); its arm, or the grid's points when the
 * schedule gives none. */
static uint32_t point_appended(struct autotune_engram_state *state,
			       uint32_t disturbance, uint32_t *phase)
{
	uint32_t arm = 0;
	const char *refusal = NULL;
	if (!autotune_engram_next(state, &arm, phase, &refusal))
		return AUTOTUNE_ENGRAM_GRID_POINTS;
	struct autotune_engram_point *const point =
		&state->points[state->point_count];
	point->arm_index = arm;
	point->phase = *phase;
	point->rotation = state->point_count++;
	point->disturbance = disturbance;
	autotune_engram_arm_of(arm, &point->arm);
	for (uint32_t level = 0; level < AUTOTUNE_ENGRAM_LEVELS; level++)
		point->levels[level].first_p50_ns =
			disturbance != AUTOTUNE_ENGRAM_QUIET ? DISTURBED_NS :
			arm == RESUMED_POINTS		     ? CHEAP_NS :
							       DEAR_NS;
	return arm;
}

/* The A D D A a slot at a time: a disturbed point is measured again in its slot, the order kept; a slot disturbed
 * twice ends it and holds the default, by name. */
static void adda_cases(struct autotune_engram_state *state)
{
	uint32_t phase = 0, arm = 0;
	const char *refusal = NULL;
	struct autotune_engram_selection selection;
	state_build(state, AUTOTUNE_ENGRAM_GRID_POINTS);
	const uint32_t challenger =
		point_appended(state, AUTOTUNE_ENGRAM_QUIET, &phase);
	const uint32_t defaults =
		point_appended(state, AUTOTUNE_ENGRAM_FAILED_READ, &phase);
	const bool again = point_appended(state, AUTOTUNE_ENGRAM_QUIET,
					  &phase) == defaults &&
			   phase == AUTOTUNE_ENGRAM_ADDA;
	const bool ordered =
		point_appended(state, AUTOTUNE_ENGRAM_QUIET, &phase) ==
			defaults &&
		point_appended(state, AUTOTUNE_ENGRAM_QUIET, &phase) ==
			challenger &&
		!autotune_engram_next(state, &arm, &phase, &refusal) &&
		refusal == NULL;
	expect(challenger == RESUMED_POINTS && defaults != challenger &&
		       again && ordered &&
		       autotune_engram_select(state, &selection, &refusal) &&
		       selection.held == AUTOTUNE_ENGRAM_ABBA_POINTS &&
		       !selection.adda_disturbed_twice &&
		       selection.slots[1] ==
			       AUTOTUNE_ENGRAM_GRID_POINTS + ADDA_REMEASURED &&
		       llround(selection.verdict.default_cost) == DEAR_NS &&
		       !selection.verdict.default_holds,
	       "a disturbed A D D A point is measured again in its slot, the order kept, and the verdict reads the "
	       "quiet one");
	static const struct {
		uint32_t quiet_before, slot;
		const char *what;
	} twice[] = {
		{ 0, 0,
		  "the challenger's A D D A slot disturbed twice ends the A D D A and holds the default, by name" },
		{ FIRST_DEFAULT_POINT + 1u, SECOND_DEFAULT_POINT,
		  "the default's A D D A slot disturbed twice ends the A D D A and holds the default, by name" }
	};
	for (size_t index = 0; index < sizeof(twice) / sizeof(twice[0]);
	     index++) {
		state_build(state, AUTOTUNE_ENGRAM_GRID_POINTS);
		for (uint32_t point = 0; point < twice[index].quiet_before;
		     point++)
			(void)point_appended(state, AUTOTUNE_ENGRAM_QUIET,
					     &phase);
		(void)point_appended(state, AUTOTUNE_ENGRAM_RETIREMENT, &phase);
		(void)point_appended(state, AUTOTUNE_ENGRAM_HOT_DRIVE, &phase);
		expect(!autotune_engram_next(state, &arm, &phase, &refusal) &&
			       refusal == NULL &&
			       autotune_engram_select(state, &selection,
						      &refusal) &&
			       selection.adda_disturbed_twice &&
			       selection.held == twice[index].slot &&
			       selection.verdict.default_holds &&
			       selection.challenger == RESUMED_POINTS,
		       twice[index].what);
	}
	state_build(state, AUTOTUNE_ENGRAM_GRID_POINTS);
	(void)point_appended(state, AUTOTUNE_ENGRAM_QUIET, &phase);
	expect(!autotune_engram_select(state, &selection, &refusal) &&
		       refusal == NULL,
	       "no selection while A D D A points remain");
}

static void schedule_cases(void)
{
	struct autotune_engram_state *const state = calloc(1, sizeof(*state));
	if (state == NULL) {
		expect(false, "the schedule cases' state is allocated");
		return;
	}
	state_build(state, AUTOTUNE_ENGRAM_GRID_POINTS);
	uint32_t seen = 0;
	for (uint32_t index = 0; index < AUTOTUNE_ENGRAM_GRID_POINTS; index++)
		seen |= 1u << state->points[index].arm_index;
	bool shuffled = false;
	for (uint32_t index = 0; index < AUTOTUNE_ENGRAM_GRID_POINTS; index++)
		shuffled = shuffled || state->points[index].arm_index != index;
	expect(state->point_count == AUTOTUNE_ENGRAM_GRID_POINTS &&
		       seen == UINT32_MAX && shuffled,
	       "the grid holds every arm once, in a seeded order");
	uint32_t arm = 0, phase = 0, challenger = 0;
	const char *refusal = NULL;
	bool adda = true;
	for (uint32_t point = 0; point < AUTOTUNE_ENGRAM_ABBA_POINTS; point++) {
		adda = adda &&
		       autotune_engram_next(state, &arm, &phase, &refusal) &&
		       phase == AUTOTUNE_ENGRAM_ADDA;
		challenger = point == 0 ? arm : challenger;
		state->points[state->point_count++].arm_index = arm;
	}
	struct autotune_engram_arm cheapest, defaults;
	autotune_engram_arm_of(challenger, &cheapest);
	autotune_engram_arm_of(
		state->points[AUTOTUNE_ENGRAM_GRID_POINTS + FIRST_DEFAULT_POINT]
			.arm_index,
		&defaults);
	expect(adda && challenger == RESUMED_POINTS &&
		       state->points[AUTOTUNE_ENGRAM_GRID_POINTS + LAST_POINT]
				       .arm_index == RESUMED_POINTS &&
		       defaults.threads_per_copy ==
			       ENGRAM_STEP_IO_THREADS_DEFAULT &&
		       defaults.spin_us == ENGRAM_STEP_SPIN_US_DEFAULT &&
		       defaults.pin == 1u &&
		       !autotune_engram_next(state, &arm, &phase, &refusal) &&
		       refusal == NULL,
	       "a quiet grid is followed by the cheapest other arm's A D D A against serving's default, then nothing");
	state_build(state, AUTOTUNE_ENGRAM_GRID_POINTS);
	state->points[DISTURBED_GRID_POINT].disturbance =
		AUTOTUNE_ENGRAM_FAILED_READ;
	const uint32_t disturbed_arm =
		state->points[DISTURBED_GRID_POINT].arm_index;
	expect(autotune_engram_next(state, &arm, &phase, &refusal) &&
		       phase == AUTOTUNE_ENGRAM_REMEASURE &&
		       arm == disturbed_arm,
	       "a disturbed grid point's arm is measured again after the grid");
	uint32_t default_place = 0;
	state_build(state, AUTOTUNE_ENGRAM_GRID_POINTS);
	for (uint32_t index = 0; index < AUTOTUNE_ENGRAM_GRID_POINTS; index++) {
		struct autotune_engram_arm grid;
		autotune_engram_arm_of(state->points[index].arm_index, &grid);
		default_place =
			grid.threads_per_copy ==
						ENGRAM_STEP_IO_THREADS_DEFAULT &&
					grid.spin_us ==
						ENGRAM_STEP_SPIN_US_DEFAULT &&
					grid.pin != 0 ?
				index :
				default_place;
	}
	state->points[default_place].disturbance = AUTOTUNE_ENGRAM_RETIREMENT;
	const bool again = autotune_engram_next(state, &arm, &phase, &refusal);
	state->points[state->point_count] = state->points[default_place];
	state->points[state->point_count++].phase = phase;
	expect(again && phase == AUTOTUNE_ENGRAM_REMEASURE &&
		       !autotune_engram_next(state, &arm, &phase, &refusal) &&
		       refusal != NULL &&
		       strstr(refusal, "default arm") != NULL,
	       "the default arm disturbed twice refuses the run");
	adda_cases(state);
	free(state);
}

/* The state at path resumed into a fresh state of the test identity, changed by change: the points it took. */
enum state_change {
	CHANGE_NONE,
	CHANGE_TRACE,
	CHANGE_PACK,
	CHANGE_KEY,
	CHANGE_SHAPE,
	CHANGES
};
static uint32_t state_resumed(const char *path,
			      struct autotune_engram_state *fresh,
			      enum state_change change)
{
	state_build(fresh, 0);
	if (change == CHANGE_TRACE)
		fresh->trace_identity[0] = 'f';
	else if (change == CHANGE_PACK)
		fresh->file_byte_count++;
	else if (change == CHANGE_KEY)
		fresh->keys[0][0] = 'f';
	else if (change == CHANGE_SHAPE)
		fresh->slots++;
	autotune_engram_state_resume(path, fresh);
	return fresh->point_count;
}

static void state_cases(const char *folder)
{
	char path[PATH_MAX];
	struct autotune_engram_state *const saved = calloc(1, sizeof(*saved));
	struct autotune_engram_state *const fresh = calloc(1, sizeof(*fresh));
	if (saved == NULL || fresh == NULL ||
	    !joined(path, sizeof(path), folder, "/profile.json.engram-state")) {
		free(saved);
		free(fresh);
		expect(false, "the state cases' state and path");
		return;
	}
	state_build(saved, RESUMED_POINTS);
	const bool written = file_write(path, saved, sizeof(*saved));
	expect(written && state_resumed(path, fresh, CHANGE_NONE) ==
				  RESUMED_POINTS,
	       "the progress of the same identity resumes");
	static const char *const what[CHANGES] = { "", "trace", "pack",
						   "copy key", "shape" };
	for (uint32_t change = CHANGE_TRACE; change < CHANGES; change++) {
		char message[SECTION_BYTES];
		(void)snprintf(message, sizeof(message),
			       "another %s starts afresh", what[change]);
		expect(written && state_resumed(path, fresh,
						(enum state_change)change) == 0,
		       message);
	}
	expect(file_write(path, saved, sizeof(*saved) - 1u) &&
		       state_resumed(path, fresh, CHANGE_NONE) == 0,
	       "a truncated state starts afresh");
	expect(file_write(path, saved, sizeof(*saved)) &&
		       chmod(path, FILE_MODE | S_IWGRP) == 0 &&
		       state_resumed(path, fresh, CHANGE_NONE) == 0,
	       "a state another user may write starts afresh");
	saved->points[1].arm_index ^= 1u;
	expect(chmod(path, FILE_MODE) == 0 &&
		       file_write(path, saved, sizeof(*saved)) &&
		       state_resumed(path, fresh, CHANGE_NONE) == 0,
	       "a state whose points are not in the schedule's order starts afresh");
	(void)unlink(path);
	free(saved);
	free(fresh);
}

/* The test trace's two token files in folder: a long case and a short one. */
static bool trace_files(const char *folder, char paths[TRACE_CASES][PATH_MAX])
{
	uint32_t long_case[CASE_LONG_TOKENS], short_case[CASE_SHORT_TOKENS];
	for (uint32_t index = 0; index < CASE_LONG_TOKENS; index++)
		long_case[index] = index % TEST_ENGRAM_PACK_VOCABULARY;
	for (uint32_t index = 0; index < CASE_SHORT_TOKENS; index++)
		short_case[index] = TEST_ENGRAM_PACK_VOCABULARY - 1u - index;
	return joined(paths[0], PATH_MAX, folder, "/long.u32") &&
	       joined(paths[1], PATH_MAX, folder, "/short.u32") &&
	       file_write(paths[0], long_case, sizeof(long_case)) &&
	       file_write(paths[1], short_case, sizeof(short_case));
}

static void trace_cases(const char *folder, const struct engram_record *engram)
{
	char paths[TRACE_CASES][PATH_MAX], bad[PATH_MAX];
	const char *const names[] = { paths[0], paths[1] };
	struct autotune_engram_trace trace;
	uint32_t failed = 0;
	const struct engram_description *const description =
		engram_description_of(engram);
	const bool opened = description != NULL && trace_files(folder, paths) &&
			    autotune_engram_trace_open(names, TRACE_CASES,
						       engram, &trace,
						       &failed) == STATUS_OK;
	uint64_t expected[TEST_ENGRAM_PACK_TABLES *
			  ENGRAM_MAXIMUM_ROWS_PER_TOKEN];
	uint32_t tokens[CASE_SHORT_TOKENS];
	for (uint32_t index = 0; index < CASE_SHORT_TOKENS; index++)
		tokens[index] = TEST_ENGRAM_PACK_VOCABULARY - 1u - index;
	expect(opened && trace.cases == TRACE_CASES &&
		       trace.tokens == CASE_LONG_TOKENS + CASE_SHORT_TOKENS &&
		       trace.references_per_token ==
			       description->table_count *
				       description->rows_per_token &&
		       engram_row_ids_entry(engram, tokens, NULL, 1,
					    expected) == STATUS_OK &&
		       memcmp(trace.row_ids +
				      (size_t)(CASE_LONG_TOKENS + 1u) *
					      trace.references_per_token,
			      expected,
			      trace.references_per_token * sizeof(uint64_t)) ==
			       0,
	       "a case's rows are its own tokens' from its own start, as engram_row_ids_entry gives them");
	uint64_t blocks = 0;
	for (uint32_t table = 0; opened && table < description->table_count;
	     table++)
		for (uint64_t block = 0; block * ENGRAM_ROWS_PER_BLOCK <
					 description->row_count[table];
		     block++) {
			bool read = false;
			for (uint32_t reference = 0;
			     reference <
			     trace.tokens * trace.references_per_token;
			     reference++)
				read = read ||
				       (reference % trace.references_per_token /
							description
								->rows_per_token ==
						table &&
					trace.row_ids[reference] /
							ENGRAM_ROWS_PER_BLOCK ==
						block);
			blocks += read ? 1u : 0u;
		}
	expect(opened && trace.unique_blocks == blocks,
	       "the trace counts the distinct blocks it reads");
	static const uint32_t decode_steps[][STEP_FIELDS] = {
		{ 0, 8 }, { 8, 8 }, { 16, 4 }, { 20, 3 }
	};
	struct autotune_engram_walk walk = { .rows_per_step = DECODE_ROWS };
	uint32_t first = 0, rows = 0, step = 0;
	bool shaped = opened;
	while (opened &&
	       autotune_engram_walk_next(&trace, &walk, &first, &rows)) {
		shaped =
			shaped &&
			step < sizeof(decode_steps) / sizeof(decode_steps[0]) &&
			first == decode_steps[step][0] &&
			rows == decode_steps[step][1];
		step++;
	}
	expect(shaped && step == sizeof(decode_steps) / sizeof(decode_steps[0]),
	       "decode steps of 8 consecutive tokens, a case's end and a short case giving short steps");
	walk = (struct autotune_engram_walk){ .first_case = 1,
					      .rows_per_step = PREFILL_ROWS };
	const bool chunked =
		opened &&
		autotune_engram_walk_next(&trace, &walk, &first, &rows) &&
		first == CASE_LONG_TOKENS && rows == CASE_SHORT_TOKENS &&
		autotune_engram_walk_next(&trace, &walk, &first, &rows) &&
		first == 0 && rows == CASE_LONG_TOKENS &&
		!autotune_engram_walk_next(&trace, &walk, &first, &rows);
	expect(chunked,
	       "a stream from the second case takes it first, each case once, in chunks of at most 1024 tokens");
	if (opened)
		autotune_engram_trace_close(&trace);
	const uint32_t outside = OUT_OF_VOCABULARY;
	const char *const bad_names[] = { paths[0], bad };
	expect(joined(bad, sizeof(bad), folder, "/bad.u32") &&
		       file_write(bad, &outside, sizeof(outside)) &&
		       autotune_engram_trace_open(bad_names, TRACE_CASES,
						  engram, &trace,
						  &failed) != STATUS_OK &&
		       failed == 1u && file_write(bad, &outside, ODD_BYTES) &&
		       autotune_engram_trace_open(bad_names, TRACE_CASES,
						  engram, &trace,
						  &failed) != STATUS_OK &&
		       failed == 1u && unlink(bad) == 0 &&
		       autotune_engram_trace_open(bad_names, TRACE_CASES,
						  engram, &trace,
						  &failed) != STATUS_OK &&
		       failed == 1u,
	       "a token outside the vocabulary, a file of no whole ids or a missing file is refused, the second "
	       "file named");
	(void)unlink(bad);
	expect(autotune_engram_rotated(TEST_ROW_MOVED, 1, TEST_ROW_COUNT) ==
			       TEST_ROW_MOVED + ENGRAM_ROWS_PER_BLOCK &&
		       autotune_engram_rotated(
			       TEST_ROW_MOVED + ENGRAM_ROWS_PER_BLOCK, 1,
			       TEST_ROW_COUNT) == TEST_ROW_MOVED &&
		       autotune_engram_rotated(TEST_ROW_COUNT -
						       ENGRAM_ROWS_PER_BLOCK,
					       1, TEST_ROW_COUNT) ==
			       TEST_ROW_COUNT - ENGRAM_ROWS_PER_BLOCK &&
		       autotune_engram_rotated(WIDE_ROW, WIDE_SHIFT,
					       WIDE_ROW_COUNT) == WIDE_MOVED,
	       "the rotation moves a row by whole blocks around its table, its place in the block kept, and leaves a "
	       "row that would land behind the last");
	(void)unlink(paths[0]);
	(void)unlink(paths[1]);
}

/* A store of two copies of the test pack in a folder of its own under TMPDIR, else /var/tmp (not on a tmpfs). */
struct test_store {
	char folder[PATH_MAX], copies[STORE_COPIES][PATH_MAX];
	struct storage_reader *reader;
	struct pack_record *pack;
	struct engram_record *engram;
	struct engram_disk *disk;
};

static bool store_open(struct test_store *store, const unsigned char *bytes)
{
	struct pack_header header;
	memcpy(&header, bytes, sizeof(header));
	const char *const temporary = getenv("TMPDIR");
	const int length = snprintf(store->folder, sizeof(store->folder),
				    "%s/anemone-engram-store-XXXXXX",
				    temporary != NULL ? temporary : "/var/tmp");
	struct statfs file_system;
	bool good = length > 0 && (size_t)length < sizeof(store->folder) &&
		    mkdtemp(store->folder) != NULL;
	if (!good)
		store->folder[0] = '\0';
	good = good && statfs(store->folder, &file_system) == 0;
	if (good && file_system.f_type == TMPFS_MAGIC) {
		(void)fprintf(
			stderr,
			"storage_autotune_engram_test: the store %s is on a tmpfs, where the step I/O does not "
			"start: set TMPDIR to a folder on a disk\n",
			store->folder);
		good = false;
	}
	const char *names[STORE_COPIES];
	for (uint32_t copy = 0; good && copy < STORE_COPIES; copy++) {
		const int named = snprintf(store->copies[copy],
					   sizeof(store->copies[copy]),
					   "%s/copy%u.anemone", store->folder,
					   copy);
		names[copy] = store->copies[copy];
		good = named > 0 &&
		       (size_t)named < sizeof(store->copies[copy]) &&
		       file_write(store->copies[copy], bytes,
				  (size_t)header.file_byte_count);
	}
	return good &&
	       storage_direct.reader_open(names, STORE_COPIES,
					  &store->reader) == STATUS_OK &&
	       pack_open_entry(&storage_direct, store->reader, &store->pack) ==
		       STATUS_OK &&
	       engram_open(&storage_direct, store->reader, store->pack,
			   &store->engram) == STATUS_OK &&
	       engram_disk_open(engram_description_of(store->engram),
				store->pack, &storage_direct, store->reader, 0,
				0, 0, 0, &store->disk) == STATUS_OK;
}

static void store_close(struct test_store *store)
{
	engram_disk_close(store->disk);
	engram_close(store->engram);
	pack_close(store->pack);
	if (store->reader != NULL)
		storage_direct.reader_close(store->reader);
	for (uint32_t copy = 0; copy < STORE_COPIES; copy++)
		if (store->copies[copy][0] != '\0')
			(void)unlink(store->copies[copy]);
	if (store->folder[0] != '\0')
		(void)rmdir(store->folder);
}

static void point_cases(const char *folder, const unsigned char *bytes)
{
	struct test_store store = { 0 };
	char paths[TRACE_CASES][PATH_MAX];
	const char *const names[] = { paths[0], paths[1] };
	struct autotune_engram_trace trace = { 0 };
	uint32_t failed = 0;
	struct autotune_engram_point *const point = calloc(1, sizeof(*point));
	const bool opened = point != NULL && store_open(&store, bytes) &&
			    trace_files(folder, paths) &&
			    autotune_engram_trace_open(names, TRACE_CASES,
						       store.engram, &trace,
						       &failed) == STATUS_OK;
	expect(opened, "the point cases' store and trace open");
	if (opened) {
		point->arm = (struct autotune_engram_arm){
			.threads_per_copy = ENGRAM_STEP_IO_THREADS_DEFAULT,
			.spin_us = ENGRAM_STEP_SPIN_US_DEFAULT,
			.pin = 1
		};
		const result_status status = autotune_engram_point_measure(
			store.disk, &trace, 0, true, point);
		bool every = status == STATUS_OK;
		for (uint32_t level = 0; level < AUTOTUNE_ENGRAM_LEVELS;
		     level++)
			every = every && point->levels[level].steps != 0 &&
				point->levels[level].first_p50_ns != 0 &&
				point->levels[level].all_p50_ns >=
					point->levels[level].first_p50_ns;
		expect(every && point->disturbance == AUTOTUNE_ENGRAM_QUIET &&
			       point->answered != 0 &&
			       point->copy_reads[0] + point->copy_reads[1] != 0,
		       "a point of serving's default arm reads every level and passes the identity check");
		/* D2 replays the trace once a stream: twice its decode steps */
		struct autotune_engram_walk walk = { .rows_per_step =
							     DECODE_ROWS };
		uint32_t first = 0, rows = 0, steps = 0;
		while (autotune_engram_walk_next(&trace, &walk, &first, &rows))
			steps++;
		expect(point->levels[DECODE_LEVEL_TWO].steps ==
			       (uint64_t)DECODE_STREAMS_TWO * steps,
		       "every stream of a level replays the whole trace once");
		memset(point, 0, sizeof(*point));
		point->arm = (struct autotune_engram_arm){
			.threads_per_copy = 1, .spin_us = 0, .pin = 0
		};
		expect(autotune_engram_point_measure(store.disk, &trace, 1,
						     true,
						     point) == STATUS_OK &&
			       point->disturbance == AUTOTUNE_ENGRAM_QUIET,
		       "a point of one free thread a copy without spin, the rows moved by a rotation, passes too");
#if CHECKING_HOOKS_ENABLED
		memset(point, 0, sizeof(*point));
		point->arm.threads_per_copy = 1;
		expect(fault_arm("engram_step_digest", 1) &&
			       autotune_engram_point_measure(
				       store.disk, &trace, DIGEST_ROTATION,
				       true, point) == STATUS_OK &&
			       fault_disarm() != 0 &&
			       point->disturbance ==
				       AUTOTUNE_ENGRAM_FAILED_READ &&
			       point->failed_reads != 0,
		       "a failed read is recovered from the other copy, and its point is disturbed");
		memset(point, 0, sizeof(*point));
		point->arm.threads_per_copy = 1;
		expect(fault_arm("engram_tune_compare", 1) &&
			       autotune_engram_point_measure(
				       store.disk, &trace, COMPARE_ROTATION,
				       true, point) == STATUS_CORRUPT_PACK &&
			       fault_disarm() != 0,
		       "a byte of the step path's rows that differs from the per-row path's refuses the point");
#endif
	}
	autotune_engram_trace_close(&trace);
	(void)unlink(paths[0]);
	(void)unlink(paths[1]);
	store_close(&store);
	free(point);
}

/* Whether this host lets a process set up an io_uring (a container's default seccomp profile refuses it): a point runs
 * serving's step I/O over it. Asked of the kernel, not of the code under test. */
static bool io_uring_allowed(void)
{
	struct io_uring_params parameters;
	memset(&parameters, 0, sizeof(parameters));
	const long ring = syscall(__NR_io_uring_setup, 1u, &parameters);
	if (ring >= 0)
		(void)close((int)ring);
	return ring >= 0;
}

/* The profile with the first copy key after member (its text, as `"engram":{`) changed in its first digit: a
 * well-formed section of another copy. */
static bool key_changed(const char *profile, const char *member)
{
	static char text[PROFILE_BYTES];
	if (!file_text(profile, text, sizeof(text)))
		return false;
	char *const found = strstr(text, member);
	char *const key = found != NULL ? strstr(found, "\"key\":\"") : NULL;
	if (key == NULL)
		return false;
	char *const digit = key + strlen("\"key\":\"");
	*digit = *digit == '0' ? '1' : '0';
	return file_write(profile, text, strlen(text));
}

/* The stage's command on stdout and stderr: stdout into out_path's file, stderr whole into text. */
static int stage_run(int count, char **arguments, const char *capture,
		     const char *out_path, char *text, size_t capacity)
{
	(void)fflush(stdout);
	const int output = open(
		out_path, O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, FILE_MODE);
	const int saved = dup(STDOUT_FILENO);
	const bool redirected = output >= 0 && saved >= 0 &&
				dup2(output, STDOUT_FILENO) == STDOUT_FILENO;
	const int status = redirected ?
				   engram_captured(count, arguments, capture,
						   text, capacity) :
				   -1;
	(void)fflush(stdout);
	if (redirected)
		(void)dup2(saved, STDOUT_FILENO);
	if (saved >= 0)
		(void)close(saved);
	if (output >= 0)
		(void)close(output);
	return status;
}

/* The stage end to end: the n-gram test pack of the model's geometry as the source in folder (the first scratch
 * device), its receipted replica on the second, in its search folder (HOME/anemone_engram_rnd/replicas), a profile
 * holding the load member; a run cut short, its resume, the section, the match, --qualify and --plan; then a replica
 * whose tables are corrupt: every read of them fails over to the source, every point is disturbed, and the run is
 * refused. */
static void stage_case(const char *folder, const char *second)
{
	unsigned char *pack = NULL;
	struct engram_record *engram = NULL;
	struct pack_header header = { 0 };
	char source[PATH_MAX], real_source[PATH_MAX], profiles[PATH_MAX],
		profile[PATH_MAX], state[PATH_MAX], home[PATH_MAX],
		replicas[PATH_MAX], replica[PATH_MAX], receipt[PATH_MAX],
		leaf[PATH_MAX], capture[PATH_MAX], printed[PATH_MAX],
		profile_option[PATH_MAX], saved_home[PATH_MAX],
		receipt_text[PATH_MAX + SECTION_BYTES],
		kept[PATH_MAX + SECTION_BYTES], paths[TRACE_CASES][PATH_MAX],
		tokens[TRACE_CASES][PATH_MAX];
	static char text[CAPTURE_LARGE], written[PROFILE_BYTES];
	const char *const old_home = getenv("HOME");
	(void)snprintf(saved_home, sizeof(saved_home), "%s",
		       old_home != NULL ? old_home : "");
	bool good = test_engram_pack_open_heads(TEST_MODEL_HEADS, &pack,
						&engram) == STATUS_OK;
	if (good)
		memcpy(&header, pack, sizeof(header));
	(void)snprintf(home, sizeof(home), "%s", second);
	char *const slash = strrchr(home, '/');
	good = good && slash != NULL && strcmp(slash + 1, SEARCH_FOLDER) == 0;
	if (slash != NULL)
		*slash = '\0';
	(void)snprintf(leaf, sizeof(leaf), "/%016" PRIx64 ".anemone",
		       header.header_digest);
	good = good &&
	       joined(source, sizeof(source), folder, "/source.anemone") &&
	       joined(profiles, sizeof(profiles), folder, "/profile") &&
	       joined(profile, sizeof(profile), profiles,
		      "/storage-profile.json") &&
	       joined(state, sizeof(state), profile, ".engram-state") &&
	       joined(replicas, sizeof(replicas), second, "/replicas") &&
	       joined(replica, sizeof(replica), replicas, leaf) &&
	       joined(receipt, sizeof(receipt), replica,
		      REPLICA_RECEIPT_SUFFIX) &&
	       joined(capture, sizeof(capture), folder, "/capture.txt") &&
	       joined(printed, sizeof(printed), folder, "/printed.txt") &&
	       joined(profile_option, sizeof(profile_option),
		      "--profile=", profile) &&
	       file_write(source, pack, (size_t)header.file_byte_count) &&
	       realpath(source, real_source) != NULL &&
	       mkdir(profiles, DIRECTORY_MODE) == 0 &&
	       mkdir(replicas, DIRECTORY_MODE) == 0 &&
	       file_write(replica, pack, (size_t)header.file_byte_count) &&
	       replicas_receipt_text(receipt_text, sizeof(receipt_text),
				     real_source, header.file_byte_count,
				     header.header_digest) &&
	       file_write(receipt, receipt_text, strlen(receipt_text)) &&
	       trace_files(folder, paths) &&
	       joined(tokens[0], sizeof(tokens[0]), "--tokens=", paths[0]) &&
	       joined(tokens[1], sizeof(tokens[1]), "--tokens=", paths[1]);
	const int kept_length =
		snprintf(kept, sizeof(kept),
			 "{\"schema\":3,\"source\":\"%s\"" LOAD_MEMBER "}\n",
			 real_source);
	good = good && kept_length > 0 && (size_t)kept_length < sizeof(kept) &&
	       file_write(profile, kept, (size_t)kept_length) &&
	       setenv("HOME", home, 1) == 0;
	expect(good,
	       "the stage case's pack, replica, receipt, tokens and profile are in place");
	char program[] = "anemone", command[] = "autotune",
	     stage[] = "--stage=engram", plan[] = "--plan",
	     qualify[] = "--qualify";
	char cut[sizeof("--points_per_run=") + sizeof("4294967295")];
	(void)snprintf(cut, sizeof(cut), "--points_per_run=%u",
		       POINTS_FIRST_RUN);
	char *first_run[] = { program,	 command,   source,	    stage,
			      tokens[0], tokens[1], profile_option, cut };
	int status = good ? stage_run((int)(sizeof(first_run) /
					    sizeof(first_run[0])),
				      first_run, capture, printed, text,
				      sizeof(text)) :
			    -1;
	expect(status == AUTOTUNE_RETRY_EXIT &&
		       occurrences(text, "storage: engram point ") ==
			       POINTS_FIRST_RUN + 1u &&
		       strstr(text, "resumable=1") != NULL &&
		       access(state, F_OK) == 0 &&
		       file_text(profile, written, sizeof(written)) &&
		       strcmp(written, kept) == 0,
	       "a run cut short measures its points after a warm-up, keeps its progress and leaves the profile");
	char *whole_run[] = { program,	 command,   source,	   stage,
			      tokens[0], tokens[1], profile_option };
	status = good ? stage_run(
				(int)(sizeof(whole_run) / sizeof(whole_run[0])),
				whole_run, capture, printed, text,
				sizeof(text)) :
			-1;
	const bool profiled = file_text(profile, written, sizeof(written));
	expect(status == EXIT_SUCCESS && profiled && access(state, F_OK) != 0 &&
		       strstr(written, LOAD_MEMBER ",\"engram\":{") != NULL &&
		       occurrences(written, "\"phase\":\"grid\"") ==
			       AUTOTUNE_ENGRAM_GRID_POINTS &&
		       occurrences(written, "\"phase\":\"adda\"") ==
			       AUTOTUNE_ENGRAM_ABBA_POINTS &&
		       occurrences(written, "\"key\":") == STORE_COPIES &&
		       strstr(written, "\"selection\":{\"winner\":{") != NULL &&
		       strstr(text, "storage: engram stage selects ") != NULL,
	       "the next run resumes, writes every point, the copies and the selection after the load member kept, and "
	       "removes its progress");
	struct storage_engram_section section;
	char reason[STORAGE_PROFILE_REASON_BYTES];
	char keys[STORE_COPIES][STORAGE_PROFILE_KEY_BYTES],
		swapped[STORE_COPIES][STORAGE_PROFILE_KEY_BYTES];
	struct local_device_description device;
	const bool keyed = storage_device_key_for(source, false, keys[0],
						  &device) == STATUS_OK &&
			   storage_device_key_for(replica, false, keys[1],
						  &device) == STATUS_OK;
	memcpy(swapped[0], keys[1], sizeof(keys[1]));
	memcpy(swapped[1], keys[0], sizeof(keys[0]));
	const bool read =
		storage_engram_section_read(profile, &section, reason);
	expect(keyed && read && section.enabled == 0 &&
		       storage_engram_section_matches(
			       &section, &header,
			       (const char (*)[STORAGE_PROFILE_KEY_BYTES])keys,
			       STORE_COPIES, section.steps, section.slots,
			       reason) &&
		       !storage_engram_section_matches(
			       &section, &header,
			       (const char (*)[STORAGE_PROFILE_KEY_BYTES])
				       swapped,
			       STORE_COPIES, section.steps, section.slots,
			       reason) &&
		       strcmp(reason, "engram_section_stale_copies") == 0,
	       "the section matches these copies' drives' keys in their order only");
	char *planned[] = { program,   command,	  source,	  stage,
			    tokens[0], tokens[1], profile_option, plan };
	status = good ? stage_run((int)(sizeof(planned) / sizeof(planned[0])),
				  planned, capture, printed, written,
				  sizeof(written)) :
			-1;
	expect(status == EXIT_SUCCESS &&
		       strstr(written, "engram store opened") == NULL &&
		       file_text(printed, text, sizeof(text)) &&
		       strstr(text,
			      "{\"valid\":true,\"stage\":\"engram\",\"copies\":2,") !=
			       NULL,
	       "--plan names both copies and opens no store, so it reads no table byte");
	char *qualified[] = { program, command,	       source,
			      stage,   profile_option, qualify };
	status = good ? stage_run(
				(int)(sizeof(qualified) / sizeof(qualified[0])),
				qualified, capture, printed, text,
				sizeof(text)) :
			-1;
	expect(status == EXIT_SUCCESS &&
		       storage_engram_section_read(profile, &section, reason) &&
		       section.enabled == 1u &&
		       file_text(profile, written, sizeof(written)) &&
		       strstr(written, LOAD_MEMBER) != NULL,
	       "--qualify enables the section that names this pack and these copies, the load member kept");
	/* The run without --stage: the copy (the replica in place, verified), then the load stage (the profile's load
	 * member is a placeholder, so it measures), then the Engram stage, whose section is current: skipped; again,
	 * both stages current; each section made stale in turn by another copy key, well formed: its stage runs again,
	 * the other skipped; --plan with --retune and no --tokens; then --retune, which measures the Engram stage again
	 * through its own command with the run's token files; then a --retune series whose load stage finished, cut
	 * before and (checking builds) inside the Engram stage, resumed to its end; then (checking builds) a series
	 * ended by an Engram stage that fails. */
	char target_option[PATH_MAX + sizeof("--target=")],
		copy_receipt[PATH_MAX], decisions[PATH_MAX], manifest[PATH_MAX],
		series[PATH_MAX] = "", floor[] = "--replica_floor_percent=0",
		workers[] = "--copy_workers=2", retune[] = "--retune";
	const bool chained =
		good &&
		joined(target_option, sizeof(target_option),
		       "--target=", replica) &&
		joined(copy_receipt, sizeof(copy_receipt), replica,
		       PREPARE_RECEIPT_SUFFIX) &&
		joined(decisions, sizeof(decisions), profile, ".copy.json") &&
		joined(manifest, sizeof(manifest), profile, ".replicas.json") &&
		joined(series, sizeof(series), profile, ".retune-series");
	char *every_stage[] = { program,	command,   source,
				target_option,	tokens[0], tokens[1],
				profile_option, floor,	   workers };
	status = chained ? stage_run((int)(sizeof(every_stage) /
					   sizeof(every_stage[0])),
				     every_stage, capture, printed, text,
				     sizeof(text)) :
			   -1;
	expect(status == EXIT_SUCCESS &&
		       strstr(text, "storage: autotune stage=copy status=ok") !=
			       NULL &&
		       strstr(text,
			      "storage: autotune stage=load runs (load_section_malformed)") !=
			       NULL &&
		       strstr(text, "storage: load stage selects ") != NULL &&
		       strstr(text,
			      "storage: autotune stage=engram skipped: its section names this pack and these "
			      "copies") != NULL &&
		       strstr(text, "storage: engram point ") == NULL &&
		       strstr(text, "storage: autotune stage=all exit=0") !=
			       NULL &&
		       storage_engram_section_read(profile, &section, reason) &&
		       section.enabled == 1u,
	       "a run without --stage copies, measures the load stage and skips the current Engram section");
	status = chained ? stage_run((int)(sizeof(every_stage) /
					   sizeof(every_stage[0])),
				     every_stage, capture, printed, text,
				     sizeof(text)) :
			   -1;
	expect(status == EXIT_SUCCESS &&
		       strstr(text,
			      "storage: autotune stage=load skipped: its section names this pack and these "
			      "copies") != NULL &&
		       strstr(text,
			      "storage: autotune stage=engram skipped: its section names") !=
			       NULL &&
		       strstr(text, "storage: load point ") == NULL &&
		       strstr(text, "storage: engram point ") == NULL,
	       "the same run again skips both stages, their sections current");
	status = chained && key_changed(profile, "\"engram\":{") ?
			 stage_run((int)(sizeof(every_stage) /
					 sizeof(every_stage[0])),
				   every_stage, capture, printed, text,
				   sizeof(text)) :
			 -1;
	expect(status == EXIT_SUCCESS &&
		       strstr(text,
			      "storage: autotune stage=load skipped: its section names") !=
			       NULL &&
		       strstr(text,
			      "storage: autotune stage=engram runs (engram_section_stale_copies)") !=
			       NULL &&
		       strstr(text, "storage: engram stage selects ") != NULL &&
		       storage_engram_section_read(profile, &section, reason) &&
		       section.enabled == 0 && access(state, F_OK) != 0,
	       "a well-formed Engram section of another copy key is not current: the stage runs again and writes a "
	       "new candidate");
	status = chained && key_changed(profile, "\"load\":{") ?
			 stage_run((int)(sizeof(every_stage) /
					 sizeof(every_stage[0])),
				   every_stage, capture, printed, text,
				   sizeof(text)) :
			 -1;
	expect(status == EXIT_SUCCESS &&
		       strstr(text,
			      "storage: autotune stage=load runs (load_section_stale_copies)") !=
			       NULL &&
		       strstr(text, "storage: load stage selects ") != NULL &&
		       strstr(text,
			      "storage: autotune stage=engram skipped: its section names") !=
			       NULL,
	       "a well-formed load section of another copy key is not current: the load stage runs again, the Engram "
	       "stage's current section skipped");
	char *planned_all[] = { program,       command,	       source,
				target_option, profile_option, floor,
				workers,       retune,	       plan };
	status = chained ? stage_run((int)(sizeof(planned_all) /
					   sizeof(planned_all[0])),
				     planned_all, capture, printed, text,
				     sizeof(text)) :
			   -1;
	expect(status == EXIT_SUCCESS &&
		       strstr(text,
			      "storage: plan: the load stage would run (--retune)") !=
			       NULL &&
		       strstr(text,
			      "storage: autotune stage=engram skipped: no --tokens") !=
			       NULL &&
		       file_text(printed, written, sizeof(written)) &&
		       strstr(written, "\"stage\":\"all\"") != NULL,
	       "--plan says the load stage would run with --retune, and without --tokens the Engram stage is skipped "
	       "by name");
	char *retuned[] = { program,   command,	  source,	  target_option,
			    tokens[0], tokens[1], profile_option, floor,
			    workers,   retune };
	status = chained ?
			 stage_run((int)(sizeof(retuned) / sizeof(retuned[0])),
				   retuned, capture, printed, text,
				   sizeof(text)) :
			 -1;
	expect(status == EXIT_SUCCESS &&
		       strstr(text,
			      "storage: autotune stage=engram runs (--retune)") !=
			       NULL &&
		       strstr(text, "storage: load stage selects ") != NULL &&
		       strstr(text, "storage: engram stage selects ") != NULL &&
		       storage_engram_section_read(profile, &section, reason) &&
		       section.enabled == 0 && access(state, F_OK) != 0 &&
		       access(series, F_OK) != 0,
	       "--retune measures both stages again; the Engram stage, given the run's token files, writes a new "
	       "candidate; the series ends with the run");
	static const char load_finished[] = "load\n";
#if CHECKING_HOOKS_ENABLED
	/* the chain: the load stage finishes and the series names it, then the lease is spent before the Engram stage
	 * (fault point autotune_stage_lease at its turn): the run exits AUTOTUNE_RETRY_EXIT and keeps the series */
	status = chained && fault_arm("autotune_stage_lease",
				      ENGRAM_STAGE_TURN) ?
			 stage_run((int)(sizeof(retuned) / sizeof(retuned[0])),
				   retuned, capture, printed, text,
				   sizeof(text)) :
			 -1;
	expect(fault_disarm() == ENGRAM_STAGE_TURN &&
		       status == AUTOTUNE_RETRY_EXIT &&
		       strstr(text, "storage: load stage selects ") != NULL &&
		       strstr(text,
			      "storage: autotune stage=engram not started: the lease is spent") !=
			       NULL &&
		       file_text(series, written, sizeof(written)) &&
		       strcmp(written, load_finished) == 0,
	       "a --retune run whose lease ends after its load stage finished exits 75 and keeps its series, which "
	       "names the load stage");
	/* then the Engram stage is cut inside (fault point engram_tune_lease before its third point): the load stage
	 * skipped by the series, the run exits AUTOTUNE_RETRY_EXIT, the Engram stage not recorded, its state kept */
	status = chained && fault_arm("engram_tune_lease", ENGRAM_POINT_CUT) ?
			 stage_run((int)(sizeof(retuned) / sizeof(retuned[0])),
				   retuned, capture, printed, text,
				   sizeof(text)) :
			 -1;
	expect(fault_disarm() == ENGRAM_POINT_CUT &&
		       status == AUTOTUNE_RETRY_EXIT &&
		       strstr(text,
			      "storage: autotune stage=load skipped: it finished in this --retune series") !=
			       NULL &&
		       strstr(text,
			      "storage: autotune stage=engram status=timed out resumable=1") !=
			       NULL &&
		       file_text(series, written, sizeof(written)) &&
		       strcmp(written, load_finished) == 0 &&
		       access(state, F_OK) == 0,
	       "a --retune run whose Engram stage is cut exits 75 and keeps its series, which still names only the "
	       "load stage, and the Engram stage's state");
	const bool resumed = chained;
#else
	/* without fault points, the series such a run leaves */
	const bool resumed = chained && file_write(series, load_finished,
						   sizeof(load_finished) - 1u);
#endif
	status = resumed ?
			 stage_run((int)(sizeof(retuned) / sizeof(retuned[0])),
				   retuned, capture, printed, text,
				   sizeof(text)) :
			 -1;
	expect(status == EXIT_SUCCESS &&
		       strstr(text,
			      "storage: autotune stage=load skipped: it finished in this --retune series") !=
			       NULL &&
		       strstr(text, "storage: load point ") == NULL &&
		       strstr(text,
			      "storage: autotune stage=engram runs (--retune)") !=
			       NULL &&
		       strstr(text, "storage: engram stage selects ") != NULL &&
		       access(series, F_OK) != 0,
	       "the same --retune command resuming its series skips the load stage that finished in it, measures the "
	       "Engram stage, and ends the series");
#if CHECKING_HOOKS_ENABLED
	/* any other exit ends the series: a new --retune series whose load stage finishes, then the Engram stage
	 * refused (fault point engram_tune_compare: the step path's rows differ) */
	status = chained && fault_arm("engram_tune_compare", 1) ?
			 stage_run((int)(sizeof(retuned) / sizeof(retuned[0])),
				   retuned, capture, printed, text,
				   sizeof(text)) :
			 -1;
	expect(fault_disarm() != 0 && status == EXIT_FAILURE &&
		       strstr(text, "storage: load stage selects ") != NULL &&
		       strstr(text,
			      "the step path's rows differ from the per-row path's") !=
			       NULL &&
		       strstr(text, "storage: autotune stage=all exit=1 ") !=
			       NULL &&
		       access(series, F_OK) != 0,
	       "a --retune run whose Engram stage fails after its load stage finished exits 1 and ends its series");
#endif
	(void)unlink(copy_receipt);
	(void)unlink(decisions);
	(void)unlink(manifest);
	(void)unlink(profile);
	if (series[0] != '\0')
		(void)unlink(series);
	char missing[PATH_MAX], missing_option[PATH_MAX + sizeof("--tokens=")];
	char *missing_run[] = { program,   command,	   source,	  stage,
				tokens[0], missing_option, profile_option };
	status = good &&
				 joined(missing, sizeof(missing), folder,
					"/missing.u32") &&
				 joined(missing_option, sizeof(missing_option),
					"--tokens=", missing) ?
			 stage_run((int)(sizeof(missing_run) /
					 sizeof(missing_run[0])),
				   missing_run, capture, printed, text,
				   sizeof(text)) :
			 -1;
	char named[PATH_MAX + SECTION_BYTES];
	(void)snprintf(named, sizeof(named),
		       "of this pack's vocabulary, within 1 MiB together: %s\n",
		       missing);
	expect(status == EXIT_FAILURE && strstr(text, named) != NULL &&
		       access(profile, F_OK) != 0,
	       "a --tokens file that does not read is refused by its own path, not the first file's");
#if CHECKING_HOOKS_ENABLED
	status = good && fault_arm("storage_flight_ring", 1) ?
			 stage_run((int)(sizeof(whole_run) /
					 sizeof(whole_run[0])),
				   whole_run, capture, printed, text,
				   sizeof(text)) :
			 -1;
	expect(fault_disarm() != 0 && status == EXIT_FAILURE &&
		       strstr(text,
			      "autotune: refused: step I/O cannot start (io_uring refused: serving keeps per-row "
			      "gathers): ") != NULL &&
		       access(profile, F_OK) != 0 && access(state, F_OK) != 0,
	       "a host whose io_uring refuses the first flight is refused by name: step I/O cannot start");
#endif
	/* every block of every table */
	const struct engram_description *const description =
		engram_description_of(engram);
	uint64_t table_blocks = 0;
	for (uint32_t table = 0;
	     description != NULL && table < description->table_count; table++)
		table_blocks += (description->row_count[table] - 1u) /
					ENGRAM_ROWS_PER_BLOCK +
				1u;
	FILE *const corrupted = fopen(replica, "r+e");
	bool damaged = corrupted != NULL && table_blocks != 0;
	for (uint64_t block = FIRST_TABLE_BLOCK;
	     damaged && block < FIRST_TABLE_BLOCK + table_blocks; block++)
		damaged =
			fseek(corrupted,
			      (long)(block * PACK_BLOCK_BYTES + CORRUPT_OFFSET),
			      SEEK_SET) == 0 &&
			fputc('x', corrupted) != EOF;
	if (corrupted != NULL)
		damaged = fclose(corrupted) == 0 && damaged;
	status = damaged ? stage_run((int)(sizeof(whole_run) /
					   sizeof(whole_run[0])),
				     whole_run, capture, printed, text,
				     sizeof(text)) :
			   -1;
	expect(status == EXIT_FAILURE &&
		       strstr(text, "disturbed: failed_read") != NULL &&
		       strstr(text,
			      "autotune: refused: the default arm was disturbed twice") !=
			       NULL &&
		       access(profile, F_OK) != 0,
	       "a replica whose blocks fail every read disturbs every point, and the run is refused");
	(void)unlink(state);
	(void)unlink(printed);
	(void)unlink(paths[0]);
	(void)unlink(paths[1]);
	(void)unlink(receipt);
	(void)unlink(replica);
	/* the copy stage's lock beside the replica, and the earlier copier's (.<leaf>.copy.lock) should it be there */
	char lock[PATH_MAX], copier_lock[PATH_MAX];
	if (joined(lock, sizeof(lock), replica, ".copy-lock"))
		(void)unlink(lock);
	if (snprintf(copier_lock, sizeof(copier_lock), "%s/.%s.copy.lock",
		     replicas, leaf + 1) > 0)
		(void)unlink(copier_lock);
	expect(rmdir(replicas) == 0 && folder_empty(second),
	       "the stage case leaves the second folder empty");
	(void)rmdir(profiles);
	(void)unlink(source);
	if (saved_home[0] != '\0')
		(void)setenv("HOME", saved_home, 1);
	engram_close(engram);
	free(pack);
}

int main(void)
{
	/* every check's line reaches the log even when a sanitizer ends the process at exit */
	(void)setvbuf(stdout, NULL, _IOLBF, 0);
	verdict_cases();
	choice_cases();
	schedule_cases();
	const char *const temporary = getenv("TMPDIR");
	char folder[PATH_MAX];
	const int length = snprintf(folder, sizeof(folder),
				    "%s/anemone-engram-stage-XXXXXX",
				    temporary != NULL ? temporary : "/tmp");
	if (length <= 0 || (size_t)length >= sizeof(folder) ||
	    mkdtemp(folder) == NULL) {
		(void)puts("FAILED: no temporary folder");
		return EXIT_FAILURE;
	}
	option_cases(folder);
	section_cases(folder);
	serving_cases(folder);
	state_cases(folder);
	unsigned char *pack = NULL;
	struct engram_record *engram = NULL;
	const bool packed = test_engram_pack_open_heads(TEST_MODEL_HEADS, &pack,
							&engram) == STATUS_OK;
	const struct engram_description *const description =
		packed ? engram_description_of(engram) : NULL;
	/* the identity check's gathers reach their real count only with the model's rows a token: 8 decode tokens of
	 * a table are 192 rows, above ENGRAM_MAXIMUM_ROWS_PER_TOKEN (the one-head pack's 8 are not) */
	expect(description != NULL &&
		       description->table_count ==
			       (uint32_t)__builtin_popcountll(
				       DS41F_ENGRAM_LAYERS) &&
		       description->rows_per_token ==
			       DS41F_ENGRAM_ROWS_PER_TOKEN,
	       "the n-gram test pack opens with the model's geometry: its two tables of 24 rows a token");
	if (packed)
		trace_cases(folder, engram);
	if (packed && io_uring_allowed())
		point_cases(folder, pack);
	else if (packed)
		(void)fputs(
			"storage_autotune_engram: the point cases require io_uring, which this host refuses\n",
			stderr);
	engram_close(engram);
	free(pack);
	const char *const first = getenv("ANEMONE_NATIVE_COPY_TEST_FIRST");
	const char *const second = getenv("ANEMONE_NATIVE_COPY_TEST_SECOND");
	char case_folder[PATH_MAX];
	const int case_length =
		first != NULL ? snprintf(case_folder, sizeof(case_folder),
					 "%s/engram-case-XXXXXX", first) :
				-1;
	if (first != NULL && second != NULL && case_length > 0 &&
	    (size_t)case_length < sizeof(case_folder) &&
	    mkdtemp(case_folder) != NULL) {
		stage_case(case_folder, second);
		expect(rmdir(case_folder) == 0,
		       "the stage case leaves its scratch folder empty");
	} else if (first != NULL && second != NULL)
		expect(false, "the stage case's scratch folder is made");
	else
		(void)fputs(
			"storage_autotune_engram: the end-to-end case requires the two-device scratch environment\n",
			stderr);
	expect(rmdir(folder) == 0, "the cases leave their folder empty");
	(void)printf("storage_autotune_engram_test: %s\n",
		     failures == 0 ? "passed" : "FAILED");
	return failures != 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
