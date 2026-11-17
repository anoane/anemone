/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * The load stage of autotune (storage_autotune_load.c) and the profile's load section (storage_profile.h):
 *   - the grid's order: each arm once a pass, no two consecutive points sharing readers or piece, every readers and
 *     piece value at the middle position of pass 1, pass 2 its reverse; each arm ranked by its two points' mean;
 *   - the A D D A D A A D verdict: the higher mean wins even inside the spread, only an exact tie goes to the simpler
 *     arm, and all eight points count;
 *   - the drives' cool bound, the cool wait's end (its bound, or the lease's end when sooner) and the page counters'
 *     JSON (a counter the kernel lacks written as null);
 *   - knob over profile over default, a parameter at a time, and no section changes nothing;
 *   - the stage's options: the copy stage's and the Engram stage's refused by name, before any file opens;
 *   - the section's read, write, replacement and qualification, every other member of the profile kept byte for byte,
 *     and a schema 2 profile or another source's refused;
 *   - in the two-device scratch environment, the stage end to end on the n-gram test pack beside a receipted replica:
 *     a state of the earlier schema is named and not resumed, a run cut short exits for a retry and the next one
 *     resumes from its progress, (checking builds) a stop signal during a warm-up or a cool wait leaves that point
 *     unmeasured for the next run, and so does the lease's end (the clock at its end during a warm-up; drives held hot
 *     through a cool wait under a short lease, the wait ending with the lease), warming up before the grid and before
 *     the confirmation, the section names both copies, every point (with its pass, CPU split, page counters and cool
 *     wait) and the selection, and matches only this pack and these copies.
 */
#include "storage_autotune.h"
#include "anemone/engram.h"
#include "anemone/hooks.h"
#include "anemone/pack.h"
#include "anemone/replicas.h"
#include "anemone/storage_exl3.h"
#include "anemone/storage_profile.h"
#include "anemone/tp_load.h"
#include "engram_pack.h"
#include "log_expectation.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CAPTURE_BYTES (256u << 10u)
#define PROFILE_BYTES (1u << 20u)
#define SECTION_BYTES 1024u
#define FIXTURE_KEY \
	"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
#define FIXTURE_DIGEST 0xaaull
#define FIXTURE_FILE_BYTES 8192ull
#define FIXTURE_READERS 2u
#define FIXTURE_REPLACED_READERS 4u
#define FIXTURE_PIECE_BYTES (8ull << 20u)
#define FIXTURE_SOURCE "/nonexistent/load-section-source.anemone"
#define KNOB_READERS 3u
#define KNOB_PIECE_BYTES (32ull << 20u)
#define POINTS_FIRST_RUN 10u
#define LOAD_FIRST_ABBA_POINT        \
	(AUTOTUNE_LOAD_GRID_POINTS + \
	 1u) /* the grid's 70 points, then the A D D A D A A D's */
/* to A1: the third run resumes inside the A D D A */
#define POINTS_SECOND_RUN (LOAD_FIRST_ABBA_POINT - POINTS_FIRST_RUN)
#define STAGE_COPIES 2u /* the source and its replica */
#define STRINGIFY_POINTS "4294967295" /* room for any count */
#define SEARCH_FOLDER "anemone_engram_rnd"
#define STAGE_WINDOW_GIB 1u /* the stage case's --window_gib */
#define ABBA_WINDOW_FACTOR \
	2u /* storage_autotune_load.c's LOAD_ABBA_WINDOW_FACTOR */
#define DECIMAL_BASE 10
#define GIB (1ull << 30u)
#define LOAD_WARMUP_PASSES_EXPECTED \
	2u /* storage_autotune_load.c's LOAD_WARMUP_PASSES */
#define ENGRAM_MEMBER ",\"engram\":{\"kept\":[1,2]}"
/* rank_cases' grid: every point at the base rate but the steady arm's two, the spiky arm's fast and slow ones and the
 * default's */
#define RANK_BASE_RATE 1.0
#define RANK_STEADY_READERS 3u
#define RANK_STEADY_PIECE 64u
#define RANK_STEADY_RATE 2.6
#define RANK_SPIKY_READERS 2u
#define RANK_SPIKY_PIECE 8u
#define RANK_SPIKY_HIGH 5.0
#define RANK_SPIKY_LOW 0.1
#define RANK_DEFAULT_PIECE 16u
#define RANK_DEFAULT_RATE 9.0
#define COOL_FIXTURE_MILLICELSIUS 30000
#define NANOSECONDS_PER_SECOND 1000000000ull
#define COOL_FIXTURE_BEGIN \
	(1000ull * NANOSECONDS_PER_SECOND) /* a wait's start, 1000 s */
#define STOPPED_COOL_CALL \
	2u /* the stopped run's cool waits: its first point's, then the stopped one's */
#define LEASE_SECONDS \
	4u /* the leased run's --seconds: the lease ends during its first point's cool wait, the drives held hot */
#define COOL_POLL_SECONDS \
	5u /* storage_autotune_load.c's LOAD_COOL_POLL_SECONDS */
#define COOL_FIXTURE_DRIVES 2u
#define WARMUPS_RESUMED_IN_GRID \
	2u /* a run resumed inside the grid: before its grid point and its first confirmation point */
#define OLD_SCHEMA_STATE 3u /* the earlier stage's state schema */
#define DIRECTORY_MODE (S_IRUSR | S_IWUSR | S_IXUSR)
#define FILE_MODE (S_IRUSR | S_IWUSR)
#define MAP_FOLDER "/dev/shm/anemone-load-map-XXXXXX"
#define MAP_MAGIC_BYTES 8u
#define MAP_RESERVED_BYTES 40u
#define MAP_SOURCES 2u /* the map itself and its one shard */
#define MAP_SHARD 1u
#define MAP_FIXTURE_BYTES 4096u
#define MEMORY_FOLDER \
	"/dev/shm/anemone-load-medium-XXXXXX" /* tmpfs: a medium serving refuses */

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
	text[0] = '\0';
	const int descriptor = open(path, O_RDONLY | O_CLOEXEC);
	const ssize_t count =
		descriptor >= 0 ?
			test_log_capture_read(descriptor, text, capacity) :
			-1;
	if (descriptor >= 0)
		(void)close(descriptor);
	return count >= 0;
}

static uint32_t occurrences(const char *text, const char *searched)
{
	uint32_t count = 0;
	for (const char *found = strstr(text, searched); found != NULL;
	     found = strstr(found + strlen(searched), searched))
		count++;
	return count;
}

/* The unsigned number right after the first anchor at or after text, else UINT64_MAX. */
static uint64_t number_after(const char *text, const char *anchor)
{
	const char *const found = text != NULL ? strstr(text, anchor) : NULL;
	return found != NULL ?
		       strtoull(found + strlen(anchor), NULL, DECIMAL_BASE) :
		       UINT64_MAX;
}

/* The autotune command with its stderr captured whole into text through the file at capture_path (made O_EXCL and
 * removed after): its exit status, or -1 when the capture fails. */
static int autotune_captured(int count, char **arguments,
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
		status = storage_autotune_cli(count, arguments);
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

/* The confirmation's rates in their order, A D D A D A A D: the challenger (A) and the default (D). */
static const double challenger_apart[AUTOTUNE_LOAD_ABBA_POINTS] = {
	10.0, 9.9, 9.95, 10.05, 9.92, 10.02, 10.03, 9.93
};
static const double challenger_within[AUTOTUNE_LOAD_ABBA_POINTS] = {
	10.0, 9.0, 11.0, 10.2, 9.5, 10.1, 10.15, 10.5
};
static const double default_ahead[AUTOTUNE_LOAD_ABBA_POINTS] = {
	10.0, 12.0, 12.1, 10.1, 12.05, 10.05, 10.08, 12.02
};
/* the load stage's first real run (15 Nov 2026): the challenger's first point was the first touch of its arena */
static const double cold_first[AUTOTUNE_LOAD_ABBA_POINTS] = {
	8.865, 34.63, 44.96, 47.61, 45.5, 46.9, 47.2, 46.1
};
/* equal at the reported precision (three decimals), apart below it */
static const double exact_tie[AUTOTUNE_LOAD_ABBA_POINTS] = {
	10.0001, 10.0002, 10.0002, 10.0001, 10.0002, 10.0001, 10.0001, 10.0002
};
/* the first four points (A D D A) favour the challenger, all eight the default */
static const double second_half[AUTOTUNE_LOAD_ABBA_POINTS] = { 10.0, 9.9, 9.9,
							       10.0, 9.9, 8.0,
							       8.0,  9.9 };

static void verdict_cases(void)
{
	struct autotune_load_verdict verdict;
	autotune_load_verdict(challenger_apart, true, &verdict);
	expect(!verdict.exact_tie && !verdict.default_wins &&
		       verdict.margin > verdict.spread,
	       "a challenger faster beyond the spread wins, though the default is simpler");
	autotune_load_verdict(challenger_within, true, &verdict);
	expect(!verdict.exact_tie && !verdict.default_wins &&
		       verdict.margin > 0.0 && verdict.margin < verdict.spread,
	       "a challenger faster inside the spread still wins (the owner's rule), the margin and spread recorded");
	autotune_load_verdict(default_ahead, false, &verdict);
	expect(!verdict.exact_tie && verdict.default_wins &&
		       verdict.margin < 0.0,
	       "a default faster than the challenger holds");
	autotune_load_verdict(exact_tie, true, &verdict);
	expect(verdict.exact_tie && verdict.default_wins,
	       "means equal at the reported precision go to the simpler arm: the default");
	autotune_load_verdict(exact_tie, false, &verdict);
	expect(verdict.exact_tie && !verdict.default_wins,
	       "means equal at the reported precision go to the simpler arm: the challenger");
	const double *const quiet[] = { challenger_apart, challenger_within,
					default_ahead, exact_tie };
	bool flagged = false;
	for (size_t index = 0; index < sizeof(quiet) / sizeof(quiet[0]);
	     index++) {
		autotune_load_verdict(quiet[index], true, &verdict);
		flagged = flagged || verdict.spread_over_bound;
	}
	expect(!flagged, "no spread within the bound is flagged");
	autotune_load_verdict(cold_first, true, &verdict);
	expect(verdict.spread_over_bound && verdict.default_wins &&
		       !verdict.exact_tie && verdict.margin < 0.0 &&
		       verdict.spread > AUTOTUNE_LOAD_SPREAD_BOUND,
	       "a spread over the bound is flagged, and the verdict is the higher mean's as before");
	autotune_load_verdict(second_half, false, &verdict);
	expect(verdict.default_wins && verdict.margin < 0.0,
	       "all eight points count: a challenger ahead over the first four loses over the eight");
}

/* The grid's order through autotune_load_arm_of: pass 1 holds each arm once, inside a pass no two consecutive points
 * share readers or piece, every readers and every piece value sits at pass 1's middle position on average, and pass 2
 * is pass 1 reversed, so each arm's two positions sum to the grid's last index. */
static void order_cases(void)
{
	uint32_t readers[AUTOTUNE_LOAD_GRID_POINTS],
		pieces[AUTOTUNE_LOAD_GRID_POINTS];
	for (uint32_t index = 0; index < AUTOTUNE_LOAD_GRID_POINTS; index++)
		autotune_load_arm_of(index, &readers[index], &pieces[index]);
	bool once = true, apart = true, mirrored = true;
	for (uint32_t index = 0; index < AUTOTUNE_LOAD_GRID_CELLS; index++) {
		for (uint32_t other = 0; other < index; other++)
			once = once && (readers[other] != readers[index] ||
					pieces[other] != pieces[index]);
		const uint32_t mirror = AUTOTUNE_LOAD_GRID_POINTS - 1u - index;
		mirrored = mirrored && readers[mirror] == readers[index] &&
			   pieces[mirror] == pieces[index];
	}
	for (uint32_t index = 1; index < AUTOTUNE_LOAD_GRID_POINTS; index++)
		apart = apart && (index == AUTOTUNE_LOAD_GRID_CELLS ||
				  (readers[index] != readers[index - 1u] &&
				   pieces[index] != pieces[index - 1u]));
	/* each value's positions in pass 1: their sum is its count times the middle position (34 / 2) */
	bool balanced = true;
	for (uint32_t index = 0; index < AUTOTUNE_LOAD_GRID_CELLS; index++) {
		uint32_t reader_sum = 0, reader_count = 0, piece_sum = 0,
			 piece_count = 0;
		for (uint32_t other = 0; other < AUTOTUNE_LOAD_GRID_CELLS;
		     other++) {
			reader_sum += readers[other] == readers[index] ? other :
									 0u;
			reader_count += readers[other] == readers[index] ? 1u :
									   0u;
			piece_sum += pieces[other] == pieces[index] ? other :
								      0u;
			piece_count += pieces[other] == pieces[index] ? 1u : 0u;
		}
		balanced =
			balanced &&
			reader_sum + reader_sum ==
				reader_count *
					(AUTOTUNE_LOAD_GRID_CELLS - 1u) &&
			piece_sum + piece_sum ==
				piece_count * (AUTOTUNE_LOAD_GRID_CELLS - 1u);
	}
	expect(once, "pass 1 runs each arm once");
	expect(apart,
	       "inside a pass no two consecutive points share readers or piece");
	expect(balanced,
	       "every readers and every piece value sits at pass 1's middle position on average");
	expect(mirrored,
	       "pass 2 is pass 1 reversed: each arm's two points average the grid's middle");
}

/* The pass 1 index of the arm readers/piece, or the cells' count when none is. */
static uint32_t arm_index(uint32_t readers, uint32_t piece_mib)
{
	for (uint32_t index = 0; index < AUTOTUNE_LOAD_GRID_CELLS; index++) {
		uint32_t found_readers = 0, found_piece = 0;
		autotune_load_arm_of(index, &found_readers, &found_piece);
		if (found_readers == readers && found_piece == piece_mib)
			return index;
	}
	return AUTOTUNE_LOAD_GRID_CELLS;
}

/* Each arm ranks by its two points' mean, not its best point; equal means go to the simpler arm. */
static void rank_cases(void)
{
	double rates[AUTOTUNE_LOAD_GRID_POINTS];
	for (uint32_t index = 0; index < AUTOTUNE_LOAD_GRID_POINTS; index++)
		rates[index] = RANK_BASE_RATE;
	const uint32_t steady = arm_index(RANK_STEADY_READERS,
					  RANK_STEADY_PIECE),
		       spiky = arm_index(RANK_SPIKY_READERS, RANK_SPIKY_PIECE),
		       fallback_index = arm_index(0, RANK_DEFAULT_PIECE);
	const uint32_t last = AUTOTUNE_LOAD_GRID_POINTS - 1u;
	bool placed = steady < AUTOTUNE_LOAD_GRID_CELLS &&
		      spiky < AUTOTUNE_LOAD_GRID_CELLS &&
		      fallback_index < AUTOTUNE_LOAD_GRID_CELLS;
	uint32_t fallback = AUTOTUNE_LOAD_GRID_CELLS;
	if (placed) {
		rates[steady] = rates[last - steady] = RANK_STEADY_RATE;
		rates[spiky] = RANK_SPIKY_HIGH;
		rates[last - spiky] = RANK_SPIKY_LOW;
		rates[fallback_index] = rates[last - fallback_index] =
			RANK_DEFAULT_RATE;
	}
	expect(placed && autotune_load_grid_rank(rates, &fallback) == steady &&
		       fallback == fallback_index,
	       "an arm ranks by its two points' mean: the steady arm beats one fast point beside a slow one, and the "
	       "faster default is not the challenger");
	if (placed)
		rates[spiky] = rates[last - spiky] = RANK_STEADY_RATE;
	expect(placed && autotune_load_grid_rank(rates, &fallback) == spiky,
	       "equal means go to the arm with fewer readers (2 against 3)");
}

/* The cool bound: every drive strictly below it; no drive read is cool. */
static void cool_cases(void)
{
	struct autotune_drives drives = { .count = COOL_FIXTURE_DRIVES };
	drives.millicelsius[0] = AUTOTUNE_LOAD_COOL_MILLICELSIUS - 1;
	drives.millicelsius[1] = COOL_FIXTURE_MILLICELSIUS;
	const bool cool = autotune_load_drives_cool(&drives);
	drives.millicelsius[1] = AUTOTUNE_LOAD_COOL_MILLICELSIUS;
	const bool hot = !autotune_load_drives_cool(&drives);
	drives.count = 0;
	expect(cool && hot && autotune_load_drives_cool(&drives),
	       "drives below the bound are cool, one at it is not, and none read is cool");
	expect(autotune_load_cool_end(COOL_FIXTURE_BEGIN, UINT64_MAX) ==
			       COOL_FIXTURE_BEGIN +
				       (uint64_t)AUTOTUNE_LOAD_COOL_SECONDS_MOST *
					       NANOSECONDS_PER_SECOND &&
		       autotune_load_cool_end(COOL_FIXTURE_BEGIN,
					      COOL_FIXTURE_BEGIN + 1u) ==
			       COOL_FIXTURE_BEGIN + 1u &&
		       autotune_load_cool_end(COOL_FIXTURE_BEGIN,
					      COOL_FIXTURE_BEGIN - 1u) ==
			       COOL_FIXTURE_BEGIN - 1u,
	       "a cool wait ends at its bound, or at the lease's end when that comes first");
	char text[SECTION_BYTES] = "";
	FILE *const stream = fmemopen(text, sizeof(text) - 1u, "w");
	const struct autotune_load_counters counters = { .user_nanoseconds = 1u,
							 .vmstat_read = 1u,
							 .vmstat = { 1u } };
	if (stream != NULL) {
		autotune_load_counters_json(stream, &counters);
		(void)fclose(stream);
	}
	expect(strstr(text,
		      "\"thp_fault_alloc\":1,\"thp_fault_fallback\":null") !=
			       NULL &&
		       strstr(text, ",\"user_nanoseconds\":1,") == text,
	       "a counter read is written as its number, one the kernel lacks as null");
}

static void choice_cases(void)
{
	const struct storage_load_section section = {
		.tp_readers_per_copy = FIXTURE_READERS,
		.tp_piece_bytes = FIXTURE_PIECE_BYTES
	};
	const struct storage_load_choice defaults = {
		.tp_readers_per_copy = KNOB_READERS,
		.tp_host_readers_per_copy = 0,
		.tp_piece_bytes = KNOB_PIECE_BYTES
	};
	struct storage_load_choice choice = defaults;
	storage_load_choose(NULL, false, false, &choice);
	expect(choice.tp_readers_per_copy == defaults.tp_readers_per_copy &&
		       choice.tp_host_readers_per_copy == 0 &&
		       choice.tp_piece_bytes == defaults.tp_piece_bytes &&
		       !choice.readers_from_profile &&
		       !choice.piece_from_profile,
	       "no applicable section leaves every value as the knob or the default gave it");
	storage_load_choose(&section, false, false, &choice);
	expect(choice.tp_host_readers_per_copy == FIXTURE_READERS &&
		       choice.tp_readers_per_copy == KNOB_READERS &&
		       choice.tp_piece_bytes == FIXTURE_PIECE_BYTES &&
		       choice.readers_from_profile && choice.piece_from_profile,
	       "an applicable section gives the host role's readers and the piece without a knob, and leaves the "
	       "whole load's readers cap at its default");
	choice = defaults;
	storage_load_choose(&section, true, false, &choice);
	expect(choice.tp_readers_per_copy == KNOB_READERS &&
		       choice.tp_host_readers_per_copy == 0 &&
		       choice.tp_piece_bytes == FIXTURE_PIECE_BYTES &&
		       !choice.readers_from_profile &&
		       choice.piece_from_profile,
	       "a readers knob caps the whole load and no host cap comes from the section, which still gives the "
	       "piece");
	choice = defaults;
	storage_load_choose(&section, false, true, &choice);
	expect(choice.tp_host_readers_per_copy == FIXTURE_READERS &&
		       choice.tp_readers_per_copy == KNOB_READERS &&
		       choice.tp_piece_bytes == KNOB_PIECE_BYTES &&
		       choice.readers_from_profile &&
		       !choice.piece_from_profile,
	       "a piece knob wins over the section, which still gives the host role's readers");
	/* SFF5: an EXL3 map's section was measured as --map_reads auto reads: DROPPED alone, DIRECT_DROPPED with
	 * replicas; an explicit other mode is another shape, auto and the same mode match */
	char reason[STORAGE_PROFILE_REASON_BYTES] = "";
	bool lone =
		storage_load_section_reads_match(1, EXL3_READ_MODES, reason) &&
		storage_load_section_reads_match(1, EXL3_READS_DROPPED,
						 reason) &&
		reason[0] == '\0';
	for (uint32_t mode = 0; mode < EXL3_READ_MODES; mode++) {
		reason[0] = '\0';
		lone = lone &&
		       (mode == EXL3_READS_DROPPED ||
			(!storage_load_section_reads_match(1, mode, reason) &&
			 strcmp(reason, "load_section_stale_shape") == 0));
	}
	expect(lone,
	       "a lone map's section matches auto and DROPPED, and any other explicit --map_reads is stale_shape");
	reason[0] = '\0';
	bool replicas =
		storage_load_section_reads_match(STAGE_COPIES, EXL3_READ_MODES,
						 reason) &&
		storage_load_section_reads_match(
			STAGE_COPIES, EXL3_READS_DIRECT_DROPPED, reason) &&
		reason[0] == '\0';
	for (uint32_t mode = 0; mode < EXL3_READ_MODES; mode++) {
		reason[0] = '\0';
		replicas = replicas &&
			   (mode == EXL3_READS_DIRECT_DROPPED ||
			    (!storage_load_section_reads_match(STAGE_COPIES,
							       mode, reason) &&
			     strcmp(reason, "load_section_stale_shape") == 0));
	}
	expect(replicas,
	       "with replicas a map's section matches auto and DIRECT_DROPPED, and any other explicit --map_reads is "
	       "stale_shape");
}

/* Each refused before the pack is opened (it does not exist), its reason and the option named. */
static void option_cases(const char *folder)
{
	char capture[PATH_MAX], text[CAPTURE_BYTES];
	if (!joined(capture, sizeof(capture), folder, "/capture.txt")) {
		expect(false, "the option cases' capture path fits");
		return;
	}
#define FOREIGN \
	"autotune: refused: the copy stage's or the Engram stage's option, not the load stage's: "
	static const struct {
		const char *option, *expected;
	} refused[] = {
		{ "--target=/nonexistent/replica.anemone",
		  FOREIGN "--target=" },
		{ "--copy_workers=2", FOREIGN "--copy_workers=2" },
		{ "--copy_hash_pairs=1", FOREIGN "--copy_hash_pairs=1" },
		{ "--tokens=/nonexistent/tokens.u32", FOREIGN "--tokens=" },
		{ "--prompt=1", "autotune: refused: " AUTOTUNE_PROMPT_RETIRED
				": --prompt=1" },
		{ "--window_gib=0",
		  "autotune: refused: --window_gib takes 1 to 1024: --window_gib=0" },
		{ "--frobnicate",
		  "autotune: refused: not an option of the load stage: --frobnicate" }
	};
	for (size_t index = 0; index < sizeof(refused) / sizeof(refused[0]);
	     index++) {
		char program[] = "anemone", command[] = "autotune",
		     source[] = "/nonexistent/load-stage.anemone",
		     stage[] = "--stage=load", option[PATH_MAX];
		(void)snprintf(option, sizeof(option), "%s",
			       refused[index].option);
		char *arguments[] = { program, command, source, stage, option };
		const int status = autotune_captured(
			(int)(sizeof(arguments) / sizeof(arguments[0])),
			arguments, capture, text, sizeof(text));
		expect(status == EXIT_FAILURE &&
			       strstr(text, refused[index].expected) != NULL,
		       refused[index].expected);
	}
	/* --help without --profile= prints the usage and succeeds (it read the profile through NULL before) */
	char program[] = "anemone", command[] = "autotune",
	     source[] = "/nonexistent/load-stage.anemone",
	     stage[] = "--stage=load", help[] = "--help";
	char *arguments[] = { program, command, source, stage, help };
	const int status = autotune_captured(
		(int)(sizeof(arguments) / sizeof(arguments[0])), arguments,
		capture, text, sizeof(text));
	expect(status == EXIT_SUCCESS &&
		       strstr(text,
			      "usage: anemone autotune PACK --stage=load") !=
			       NULL &&
		       strstr(text, "refused") == NULL,
	       "--stage=load --help without --profile= prints the usage alone and exits 0");
}

static bool section_text(char *output, size_t capacity, uint32_t readers,
			 uint64_t piece_bytes, const char *key,
			 const char *split)
{
	const int length = snprintf(
		output, capacity,
		"{\"schema\":%u,\"enabled\":0,\"qualification\":\"candidate\",\"created_unix\":1,"
		"\"pack\":{\"header_digest\":\"%016" PRIx64
		"\",\"file_byte_count\":%" PRIu64 "},"
		"\"copies\":[{\"key\":\"%s\",\"path\":\"/x\",\"filesystem\":\"xfs\"}],"
		"\"tp\":{\"readers_per_copy\":%" PRIu32
		",\"piece_bytes\":%" PRIu64
		",\"host_threads\":%u,\"split\":\"%s\"}}",
		STORAGE_LOAD_SECTION_SCHEMA, (uint64_t)FIXTURE_DIGEST,
		(uint64_t)FIXTURE_FILE_BYTES, key, readers, piece_bytes,
		TP_LOAD_HOST_THREADS_DEFAULT, split);
	return length > 0 && (size_t)length < capacity;
}

static bool section_written(const char *profile, uint32_t readers,
			    uint64_t piece_bytes, const char *key,
			    const char **refusal)
{
	char section[SECTION_BYTES];
	return section_text(section, sizeof(section), readers, piece_bytes, key,
			    tp_load_split_names[TP_LOAD_SPLIT_STEAL]) &&
	       storage_load_section_write(profile, FIXTURE_SOURCE, section,
					  strlen(section),
					  refusal) == STATUS_OK;
}

static void section_cases(const char *folder)
{
	char profile[PATH_MAX], text[PROFILE_BYTES], before[PROFILE_BYTES];
	char reason[STORAGE_PROFILE_REASON_BYTES];
	struct storage_load_section section;
	const char *refusal = NULL;
	if (!joined(profile, sizeof(profile), folder,
		    "/storage-profile.json")) {
		expect(false, "the section cases' profile path fits");
		return;
	}
	expect(!storage_load_section_read(profile, &section, reason) &&
		       strcmp(reason, "missing") == 0,
	       "no profile: no section, reason missing");
	expect(section_written(profile, FIXTURE_READERS, FIXTURE_PIECE_BYTES,
			       FIXTURE_KEY, &refusal) &&
		       storage_load_section_read(profile, &section, reason) &&
		       section.enabled == 0 && section.copy_count == 1u &&
		       section.tp_readers_per_copy == FIXTURE_READERS &&
		       section.tp_piece_bytes == FIXTURE_PIECE_BYTES &&
		       section.file_byte_count == FIXTURE_FILE_BYTES &&
		       section.tp_host_threads ==
			       TP_LOAD_HOST_THREADS_DEFAULT &&
		       section.tp_split == TP_LOAD_SPLIT_STEAL &&
		       strcmp(section.keys[0], FIXTURE_KEY) == 0 &&
		       strcmp(section.source, FIXTURE_SOURCE) == 0,
	       "a new profile is a schema 3 profile of the source with the section as written");
	static const char kept[] = "{\"schema\":3,\"source\":\"" FIXTURE_SOURCE
				   "\"" ENGRAM_MEMBER "}\n";
	expect(file_write(profile, kept, sizeof(kept) - 1u) &&
		       section_written(profile, FIXTURE_READERS,
				       FIXTURE_PIECE_BYTES, FIXTURE_KEY,
				       &refusal) &&
		       file_text(profile, text, sizeof(text)) &&
		       strstr(text, ENGRAM_MEMBER ",\"load\":{") != NULL,
	       "a section is added after every other member, which stays byte for byte");
	expect(section_written(profile, FIXTURE_REPLACED_READERS,
			       FIXTURE_PIECE_BYTES, FIXTURE_KEY, &refusal) &&
		       file_text(profile, text, sizeof(text)) &&
		       strstr(text, ENGRAM_MEMBER ",\"load\":{") != NULL &&
		       occurrences(text, "\"load\":") == 1u &&
		       storage_load_section_read(profile, &section, reason) &&
		       section.tp_readers_per_copy == FIXTURE_REPLACED_READERS,
	       "a section written again replaces the old one in place, the rest kept");
	expect(storage_load_section_qualify(profile, FIXTURE_SOURCE,
					    &refusal) == STATUS_OK &&
		       storage_load_section_read(profile, &section, reason) &&
		       section.enabled == 1u &&
		       file_text(profile, text, sizeof(text)) &&
		       strstr(text,
			      "\"enabled\":1,\"qualification\":\"qualified\"") !=
			       NULL &&
		       strstr(text, ENGRAM_MEMBER) != NULL,
	       "--qualify enables the candidate and changes nothing else");
	expect(storage_load_section_qualify(profile, FIXTURE_SOURCE,
					    &refusal) != STATUS_OK &&
		       refusal != NULL &&
		       strstr(refusal, "no candidate") != NULL,
	       "a qualified section is no candidate to qualify again");
	const struct pack_header stale = { .header_digest = FIXTURE_DIGEST + 1u,
					   .file_byte_count =
						   FIXTURE_FILE_BYTES };
	const struct pack_header same = { .header_digest = FIXTURE_DIGEST,
					  .file_byte_count =
						  FIXTURE_FILE_BYTES };
	expect(!storage_load_section_matches(&section, &stale, NULL, 0, false,
					     TP_LOAD_HOST_THREADS_DEFAULT,
					     TP_LOAD_SPLIT_STEAL, reason) &&
		       strcmp(reason, "load_section_stale_pack") == 0,
	       "a section of another pack is stale");
	expect(!storage_load_section_matches(&section, &same, NULL, 0, false,
					     TP_LOAD_HOST_THREADS_DEFAULT,
					     TP_LOAD_SPLIT_STEAL, reason) &&
		       strcmp(reason, "load_section_stale_copies") == 0,
	       "a section of other copies is stale");
	/* the shape after the copies (no copy here, so no drive needs a key): host threads and split as measured */
	struct storage_load_section shaped = section;
	shaped.copy_count = 0;
	expect(storage_load_section_matches(&shaped, &same, NULL, 0, false,
					    TP_LOAD_HOST_THREADS_DEFAULT,
					    TP_LOAD_SPLIT_STEAL, reason) &&
		       !storage_load_section_matches(
			       &shaped, &same, NULL, 0, false,
			       TP_LOAD_HOST_THREADS_DEFAULT + 1u,
			       TP_LOAD_SPLIT_STEAL, reason) &&
		       strcmp(reason, "load_section_stale_shape") == 0 &&
		       !storage_load_section_matches(
			       &shaped, &same, NULL, 0, false,
			       TP_LOAD_HOST_THREADS_DEFAULT,
			       TP_LOAD_SPLIT_FIXED, reason) &&
		       strcmp(reason, "load_section_stale_shape") == 0,
	       "a section measured with other host threads or another split is stale");
	expect(storage_load_section_writable(profile, FIXTURE_SOURCE,
					     &refusal) == STATUS_OK &&
		       refusal == NULL,
	       "the stage's own schema 3 profile passes the read-only check");
	static const char combined[] =
		"{\"schema\":2,\"source\":\"" FIXTURE_SOURCE "\"}\n";
	expect(file_write(profile, combined, sizeof(combined) - 1u) &&
		       storage_load_section_writable(profile, FIXTURE_SOURCE,
						     &refusal) != STATUS_OK &&
		       refusal != NULL &&
		       strstr(refusal, "no stage profile") != NULL &&
		       !section_written(profile, FIXTURE_READERS,
					FIXTURE_PIECE_BYTES, FIXTURE_KEY,
					&refusal) &&
		       refusal != NULL &&
		       strstr(refusal, "no stage profile") != NULL &&
		       file_text(profile, before, sizeof(before)) &&
		       strcmp(before, combined) == 0 &&
		       !storage_load_section_read(profile, &section, reason) &&
		       strcmp(reason, "not_a_stage_profile") == 0,
	       "a schema 2 profile is left as it is, and holds no load section for serving");
	static const char other[] =
		"{\"schema\":3,\"source\":\"/nonexistent/other.anemone\"}\n";
	expect(file_write(profile, other, sizeof(other) - 1u) &&
		       !section_written(profile, FIXTURE_READERS,
					FIXTURE_PIECE_BYTES, FIXTURE_KEY,
					&refusal) &&
		       refusal != NULL &&
		       strstr(refusal, "another source") != NULL &&
		       storage_load_section_writable(profile, FIXTURE_SOURCE,
						     &refusal) != STATUS_OK &&
		       refusal != NULL &&
		       strstr(refusal, "another source") != NULL,
	       "another source's profile is refused, by the read-only check too");
	(void)unlink(profile);
	expect(storage_load_section_writable(profile, FIXTURE_SOURCE,
					     &refusal) == STATUS_OK &&
		       refusal == NULL,
	       "no profile passes the read-only check");
	expect(section_written(profile, FIXTURE_READERS, PACK_BLOCK_BYTES + 1u,
			       FIXTURE_KEY, &refusal) &&
		       !storage_load_section_read(profile, &section, reason) &&
		       strcmp(reason, "load_section_malformed") == 0,
	       "a piece of no whole pack blocks is malformed");
	(void)unlink(profile);
	expect(section_written(profile, FIXTURE_READERS, FIXTURE_PIECE_BYTES,
			       "not-a-key", &refusal) &&
		       !storage_load_section_read(profile, &section, reason) &&
		       strcmp(reason, "load_section_malformed") == 0,
	       "a copy key that is no digest is malformed");
	(void)unlink(profile);
	char sideways[SECTION_BYTES];
	expect(section_text(sideways, sizeof(sideways), FIXTURE_READERS,
			    FIXTURE_PIECE_BYTES, FIXTURE_KEY, "sideways") &&
		       storage_load_section_write(profile, FIXTURE_SOURCE,
						  sideways, strlen(sideways),
						  &refusal) == STATUS_OK &&
		       !storage_load_section_read(profile, &section, reason) &&
		       strcmp(reason, "load_section_malformed") == 0,
	       "a split of no known name is malformed");
	(void)unlink(profile);
}

/* Refusals before any point, on the test pack on tmpfs: a schema 2 profile (the combined tuning's) is refused before
 * the copies are described, its bytes kept; then the source's medium (serving refuses tmpfs) is refused by name, not
 * as a pack with no host-only extents. */
static void early_refusal_cases(const char *folder)
{
	unsigned char *pack = NULL;
	struct engram_record *engram = NULL;
	struct pack_header header = { 0 };
	char memory[] = MEMORY_FOLDER, source[PATH_MAX], capture[PATH_MAX],
	     printed[PATH_MAX], profile[PATH_MAX], profile_option[PATH_MAX];
	static char text[CAPTURE_BYTES], kept[PROFILE_BYTES],
		plan_line[CAPTURE_BYTES];
	bool good = test_engram_pack_open(&pack, &engram) == STATUS_OK;
	if (good)
		memcpy(&header, pack, sizeof(header));
	const bool made = good && mkdtemp(memory) != NULL;
	good = made &&
	       joined(profile, sizeof(profile), folder, "/medium.json") &&
	       joined(profile_option, sizeof(profile_option),
		      "--profile=", profile) &&
	       joined(source, sizeof(source), memory, "/source.anemone") &&
	       joined(capture, sizeof(capture), folder,
		      "/medium-capture.txt") &&
	       joined(printed, sizeof(printed), folder, "/medium-plan.txt") &&
	       file_write(source, pack, (size_t)header.file_byte_count);
	char program[] = "anemone", command[] = "autotune",
	     stage[] = "--stage=load", plan[] = "--plan";
	char *arguments[] = { program, command,	       source,
			      stage,   profile_option, plan };
	static const char combined[] = "{\"schema\":2,\"source\":\"/x\"}\n";
	/* the plan's line on stdout, through the file at printed */
	(void)fflush(stdout);
	const int plan_output =
		good ? open(printed, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC,
			    FILE_MODE) :
		       -1;
	const int saved_output = dup(STDOUT_FILENO);
	const bool redirected = plan_output >= 0 && saved_output >= 0 &&
				dup2(plan_output, STDOUT_FILENO) ==
					STDOUT_FILENO;
	const int combined_status =
		redirected && file_write(profile, combined,
					 sizeof(combined) - 1u) ?
			autotune_captured(
				(int)(sizeof(arguments) / sizeof(arguments[0])),
				arguments, capture, text, sizeof(text)) :
			-1;
	(void)fflush(stdout);
	if (redirected)
		(void)dup2(saved_output, STDOUT_FILENO);
	if (saved_output >= 0)
		(void)close(saved_output);
	if (plan_output >= 0)
		(void)close(plan_output);
	expect(file_text(printed, plan_line, sizeof(plan_line)) &&
		       strcmp(plan_line,
			      "{\"valid\":false,\"stage\":\"load\"}\n") == 0,
	       "the plan prints valid:false for a profile the stage would refuse");
	(void)unlink(printed);
	expect(combined_status == EXIT_FAILURE &&
		       strstr(text,
			      "autotune: refused: the profile is no stage profile (schema 3)") !=
			       NULL &&
		       strstr(text, "the copy's medium") == NULL &&
		       file_text(profile, kept, sizeof(kept)) &&
		       strcmp(kept, combined) == 0,
	       "the combined tuning's schema 2 profile is refused before any copy is described, and kept");
	(void)unlink(profile);
	const int status = good ? autotune_captured((int)(sizeof(arguments) /
							  sizeof(arguments[0])),
						    arguments, capture, text,
						    sizeof(text)) :
				  -1;
	expect(status == EXIT_FAILURE &&
		       strstr(text,
			      "autotune: refused: the copy's medium is refused "
			      "(virtual_filesystem_or_device): ") != NULL &&
		       strstr(text, "no host-only extents") == NULL,
	       "a source on tmpfs is refused by its medium's name, not as a pack with no host-only extents");
	(void)unlink(source);
	if (made)
		(void)rmdir(memory);
	engram_close(engram);
	free(pack);
}

/* An .exl3map of MAP_FIXTURE_BYTES logical bytes, all from the one shard (storage_exl3.h's layout). */
struct map_record {
	unsigned char magic[MAP_MAGIC_BYTES];
	uint64_t bytes;
	uint32_t sources, segments;
	unsigned char reserved[MAP_RESERVED_BYTES];
	char paths[MAP_SOURCES][EXL3_MAP_PATH_BYTES];
	uint64_t segment_offset, segment_bytes, source_offset;
	uint32_t source, segment_reserved;
};
_Static_assert(sizeof(struct map_record) ==
		       EXL3_MAP_HEADER_BYTES +
			       MAP_SOURCES * EXL3_MAP_PATH_BYTES +
			       EXL3_MAP_SEGMENT_BYTES,
	       "the map fixture's layout");

/* A section keys an EXL3 map by its data drive (storage_device_path), as the stage does: a map file on tmpfs, a
 * medium that has no key, over a shard in folder (on the first scratch device) matches a section keyed by the shard's
 * drive. */
static void exl3_key_case(const char *folder)
{
	char memory[] = MAP_FOLDER, map[PATH_MAX], shard[PATH_MAX];
	char reason[STORAGE_PROFILE_REASON_BYTES];
	static const unsigned char shard_bytes[MAP_FIXTURE_BYTES];
	static struct map_record record = { .magic = { 'A', 'N', 'E', 'X', 'L',
						       '3', 0, 0 },
					    .bytes = MAP_FIXTURE_BYTES,
					    .sources = MAP_SOURCES,
					    .segments = 1,
					    .segment_bytes = MAP_FIXTURE_BYTES,
					    .source = MAP_SHARD };
	const bool made = mkdtemp(memory) != NULL;
	bool good = made &&
		    joined(map, sizeof(map), memory, "/model.exl3map") &&
		    joined(shard, sizeof(shard), folder, "/shard.bin") &&
		    strlen(shard) < EXL3_MAP_PATH_BYTES &&
		    file_write(shard, shard_bytes, sizeof(shard_bytes));
	if (good)
		memcpy(record.paths[MAP_SHARD], shard, strlen(shard) + 1u);
	good = good && file_write(map, &record, sizeof(record));
	struct storage_load_section section = {
		.copy_count = 1,
		.file_byte_count = FIXTURE_FILE_BYTES,
		.tp_host_threads = TP_LOAD_HOST_THREADS_DEFAULT,
		.tp_split = TP_LOAD_SPLIT_STEAL
	};
	(void)snprintf(section.pack_identity, sizeof(section.pack_identity),
		       "%016" PRIx64, (uint64_t)FIXTURE_DIGEST);
	const struct pack_header header = { .header_digest = FIXTURE_DIGEST,
					    .file_byte_count =
						    FIXTURE_FILE_BYTES };
	const char *const copies[] = { map };
	struct local_device_description device;
	expect(good &&
		       storage_device_key_for(shard, false, section.keys[0],
					      &device) == STATUS_OK &&
		       storage_load_section_matches(
			       &section, &header, copies, 1, false,
			       TP_LOAD_HOST_THREADS_DEFAULT,
			       TP_LOAD_SPLIT_STEAL, reason),
	       "an EXL3 map's section is keyed by its data drive, not by the map file's");
	(void)unlink(map);
	(void)unlink(shard);
	if (made)
		(void)rmdir(memory);
}

/* SFF4, SF9's stage side: the load stage's plan over an EXL3 map on tmpfs (a medium with no key) whose one segment is
 * the n-gram test pack, a shard in folder (on the first scratch device), with no replica to find: the copy is keyed by
 * its data drive (storage_device_path), so the plan is valid with one copy. */
static void exl3_plan_case(const char *folder)
{
	unsigned char *pack = NULL;
	struct engram_record *engram = NULL;
	struct pack_header header = { 0 };
	char memory[] = MAP_FOLDER, map[PATH_MAX], shard[PATH_MAX],
	     home[PATH_MAX], profile[PATH_MAX], profile_option[PATH_MAX],
	     capture[PATH_MAX], printed[PATH_MAX], saved_home[PATH_MAX];
	static char text[CAPTURE_BYTES], plan_line[CAPTURE_BYTES];
	static struct map_record record = { .magic = { 'A', 'N', 'E', 'X', 'L',
						       '3', 0, 0 },
					    .sources = MAP_SOURCES,
					    .segments = 1,
					    .source = MAP_SHARD };
	const char *const old_home = getenv("HOME");
	(void)snprintf(saved_home, sizeof(saved_home), "%s",
		       old_home != NULL ? old_home : "");
	bool good = test_engram_pack_open(&pack, &engram) == STATUS_OK;
	if (good)
		memcpy(&header, pack, sizeof(header));
	const bool made = good && mkdtemp(memory) != NULL;
	good = made && joined(map, sizeof(map), memory, "/model.exl3map") &&
	       joined(shard, sizeof(shard), folder, "/map-shard.bin") &&
	       strlen(shard) < EXL3_MAP_PATH_BYTES &&
	       joined(home, sizeof(home), folder, "/map-home") &&
	       joined(profile, sizeof(profile), folder, "/map-profile.json") &&
	       joined(profile_option, sizeof(profile_option),
		      "--profile=", profile) &&
	       joined(capture, sizeof(capture), folder, "/map-capture.txt") &&
	       joined(printed, sizeof(printed), folder, "/map-plan.txt") &&
	       file_write(shard, pack, (size_t)header.file_byte_count) &&
	       mkdir(home, DIRECTORY_MODE) == 0;
	record.bytes = record.segment_bytes = header.file_byte_count;
	if (good)
		memcpy(record.paths[MAP_SHARD], shard, strlen(shard) + 1u);
	good = good && file_write(map, &record, sizeof(record)) &&
	       setenv("HOME", home, 1) == 0;
	char program[] = "anemone", command[] = "autotune",
	     stage[] = "--stage=load", plan[] = "--plan";
	char *arguments[] = {
		program, command, map, stage, profile_option, plan
	};
	(void)fflush(stdout);
	const int plan_output =
		good ? open(printed, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC,
			    FILE_MODE) :
		       -1;
	const int saved_output = dup(STDOUT_FILENO);
	const bool redirected = plan_output >= 0 && saved_output >= 0 &&
				dup2(plan_output, STDOUT_FILENO) ==
					STDOUT_FILENO;
	const int status =
		redirected ?
			autotune_captured(
				(int)(sizeof(arguments) / sizeof(arguments[0])),
				arguments, capture, text, sizeof(text)) :
			-1;
	(void)fflush(stdout);
	if (redirected)
		(void)dup2(saved_output, STDOUT_FILENO);
	if (saved_output >= 0)
		(void)close(saved_output);
	if (plan_output >= 0)
		(void)close(plan_output);
	expect(good && status == EXIT_SUCCESS &&
		       file_text(printed, plan_line, sizeof(plan_line)) &&
		       strncmp(plan_line,
			       "{\"valid\":true,\"stage\":\"load\",\"copies\":1,",
			       strlen("{\"valid\":true,\"stage\":\"load\",\"copies\":1,")) ==
			       0 &&
		       strstr(text, "the copy's medium") == NULL,
	       "the stage plans over an EXL3 map on tmpfs with one copy, keyed by its shard's drive");
	(void)setenv("HOME", saved_home, 1);
	(void)unlink(printed);
	(void)unlink(map);
	(void)unlink(shard);
	(void)rmdir(home);
	if (made)
		(void)rmdir(memory);
	engram_close(engram);
	free(pack);
}

/* The stage end to end: the n-gram test pack as the source in folder (on the first scratch device), its receipted
 * replica on the second, in its search folder (HOME/anemone_engram_rnd/replicas, HOME being second's parent), and a
 * schema 3 profile holding another stage's member. */
static void stage_case(const char *folder, const char *second)
{
	unsigned char *pack = NULL;
	struct engram_record *engram = NULL;
	struct pack_header header = { 0 };
	char source[PATH_MAX], real_source[PATH_MAX], profiles[PATH_MAX],
		profile[PATH_MAX], state[PATH_MAX], home[PATH_MAX],
		replicas[PATH_MAX], replica[PATH_MAX], receipt[PATH_MAX],
		leaf[PATH_MAX], capture[PATH_MAX], profile_option[PATH_MAX],
		receipt_text[PATH_MAX + SECTION_BYTES];
	static char text[CAPTURE_BYTES], written[PROFILE_BYTES];
	const char *const old_home = getenv("HOME");
	char saved_home[PATH_MAX];
	(void)snprintf(saved_home, sizeof(saved_home), "%s",
		       old_home != NULL ? old_home : "");
	bool good = test_engram_pack_open(&pack, &engram) == STATUS_OK;
	if (good)
		memcpy(&header, pack, sizeof(header));
	(void)snprintf(home, sizeof(home), "%s", second);
	char *const slash = strrchr(home, '/');
	good = good && slash != NULL && strcmp(slash + 1, SEARCH_FOLDER) == 0;
	expect(good,
	       "the second scratch folder is a replica search folder (anemone_engram_rnd)");
	if (slash != NULL)
		*slash = '\0';
	(void)snprintf(leaf, sizeof(leaf), "/%016" PRIx64 ".anemone",
		       header.header_digest);
	good = good &&
	       joined(source, sizeof(source), folder, "/source.anemone") &&
	       joined(profiles, sizeof(profiles), folder, "/profile") &&
	       joined(profile, sizeof(profile), profiles,
		      "/storage-profile.json") &&
	       joined(state, sizeof(state), profile, ".load-state") &&
	       joined(replicas, sizeof(replicas), second, "/replicas") &&
	       joined(replica, sizeof(replica), replicas, leaf) &&
	       joined(receipt, sizeof(receipt), replica,
		      REPLICA_RECEIPT_SUFFIX) &&
	       joined(capture, sizeof(capture), folder, "/capture.txt") &&
	       joined(profile_option, sizeof(profile_option),
		      "--profile=", profile);
	good = good &&
	       file_write(source, pack, (size_t)header.file_byte_count) &&
	       realpath(source, real_source) != NULL &&
	       mkdir(profiles, DIRECTORY_MODE) == 0 &&
	       mkdir(replicas, DIRECTORY_MODE) == 0 &&
	       file_write(replica, pack, (size_t)header.file_byte_count) &&
	       replicas_receipt_text(receipt_text, sizeof(receipt_text),
				     real_source, header.file_byte_count,
				     header.header_digest) &&
	       file_write(receipt, receipt_text, strlen(receipt_text));
	char kept[PATH_MAX + SECTION_BYTES];
	const int kept_length =
		snprintf(kept, sizeof(kept),
			 "{\"schema\":3,\"source\":\"%s\"" ENGRAM_MEMBER "}\n",
			 real_source);
	good = good && kept_length > 0 && (size_t)kept_length < sizeof(kept) &&
	       file_write(profile, kept, (size_t)kept_length) &&
	       setenv("HOME", home, 1) == 0;
	const uint32_t old_state[] = { OLD_SCHEMA_STATE, STAGE_COPIES };
	good = good && file_write(state, old_state, sizeof(old_state)) &&
	       chmod(state, FILE_MODE) == 0;
	expect(good,
	       "the stage case's pack, replica, receipt, profile and an earlier schema's state are in place");
	char program[] = "anemone", command[] = "autotune",
	     stage[] = "--stage=load", window[] = "--window_gib=1",
	     qualify[] = "--qualify";
	char cut[sizeof("--points_per_run=") + sizeof(STRINGIFY_POINTS)];
	(void)snprintf(cut, sizeof(cut), "--points_per_run=%u",
		       POINTS_FIRST_RUN);
	char *first_run[] = { program,	      command, source, stage,
			      profile_option, window,  cut };
	int status = good ? autotune_captured((int)(sizeof(first_run) /
						    sizeof(first_run[0])),
					      first_run, capture, text,
					      sizeof(text)) :
			    -1;
	expect(strstr(text, " is schema 3, not 4: not resumed") != NULL,
	       "a state of the earlier schema is named and not resumed");
	const char *grid_warmed =
		strstr(text, "storage: load warm-up before the grid: ");
	expect(grid_warmed != NULL &&
		       occurrences(text, "storage: load warm-up ") == 1u &&
		       grid_warmed < strstr(text, "storage: load point 1/"),
	       "unmeasured arenas are touched before the grid's first point");
	expect(status == AUTOTUNE_RETRY_EXIT &&
		       occurrences(text, "storage: load point ") ==
			       POINTS_FIRST_RUN &&
		       strstr(text, "storage: autotune stage=load status=") !=
			       NULL &&
		       strstr(text, "resumable=1") != NULL &&
		       access(state, F_OK) == 0 &&
		       file_text(profile, written, sizeof(written)) &&
		       strcmp(written, kept) == 0,
	       "a run cut short measures its points, keeps its progress, exits for a retry and leaves the profile");
	/* checking builds: a stop signal during the grid's warm-up, then during the second point's cool wait (fault
	   points); neither stopped point is measured, and the next run measures it */
	uint32_t stopped_measured = 0;
	char abba_point[sizeof("storage: load point /") +
			sizeof(STRINGIFY_POINTS)];
#if CHECKING_HOOKS_ENABLED
	char stopped_point[sizeof("storage: load stopped before point /") +
			   sizeof(STRINGIFY_POINTS)];
	char *stopped_run[] = { program, command,	 source,
				stage,	 profile_option, window };
	const int stopped_words =
		(int)(sizeof(stopped_run) / sizeof(stopped_run[0]));
	status = good && fault_arm("load_warmup_stop", 1u) ?
			 autotune_captured(stopped_words, stopped_run, capture,
					   text, sizeof(text)) :
			 -1;
	(void)snprintf(stopped_point, sizeof(stopped_point),
		       "storage: load stopped before point %u/",
		       POINTS_FIRST_RUN + 1u);
	expect(fault_disarm() == 1u && status == AUTOTUNE_RETRY_EXIT &&
		       occurrences(text, "storage: load point ") == 0 &&
		       occurrences(text, "storage: load warm-up ") == 0 &&
		       strstr(text, stopped_point) != NULL,
	       "a stop during the warm-up ends the run before its point is measured, the warm-up unrecorded");
	status = good && fault_arm("load_cool_stop", STOPPED_COOL_CALL) ?
			 autotune_captured(stopped_words, stopped_run, capture,
					   text, sizeof(text)) :
			 -1;
	(void)snprintf(stopped_point, sizeof(stopped_point),
		       "storage: load stopped before point %u/",
		       POINTS_FIRST_RUN + STOPPED_COOL_CALL);
	expect(fault_disarm() == STOPPED_COOL_CALL &&
		       status == AUTOTUNE_RETRY_EXIT &&
		       occurrences(text, "storage: load point ") == 1u &&
		       strstr(text, stopped_point) != NULL,
	       "a stop during a point's cool wait leaves that point unmeasured and the run exits for a retry");
	stopped_measured = STOPPED_COOL_CALL - 1u;
	/* the lease's end with no stop signal: the clock at the lease's end during the warm-up, then drives held hot
	   through the first point's cool wait under a lease of LEASE_SECONDS; neither point is measured */
	(void)snprintf(stopped_point, sizeof(stopped_point),
		       "storage: load stopped before point %u/",
		       POINTS_FIRST_RUN + stopped_measured + 1u);
	status = good && fault_arm("load_warmup_lease_end", 1u) ?
			 autotune_captured(stopped_words, stopped_run, capture,
					   text, sizeof(text)) :
			 -1;
	expect(fault_disarm() == 1u && status == AUTOTUNE_RETRY_EXIT &&
		       occurrences(text, "storage: load point ") == 0 &&
		       occurrences(text, "storage: load warm-up ") == 0 &&
		       strstr(text, stopped_point) != NULL,
	       "the lease's end during the warm-up ends the run before its point is measured, the warm-up unrecorded");
	char lease[sizeof("--seconds=") + sizeof(STRINGIFY_POINTS)];
	(void)snprintf(lease, sizeof(lease), "--seconds=%u", LEASE_SECONDS);
	char *leased_run[] = { program,	       command, source, stage,
			       profile_option, window,	lease };
	status = good && fault_arm("load_cool_hot", 1u) ?
			 autotune_captured((int)(sizeof(leased_run) /
						 sizeof(leased_run[0])),
					   leased_run, capture, text,
					   sizeof(text)) :
			 -1;
	const uint64_t waited = number_after(strstr(text, stopped_point),
					     "after a cool wait of ");
	expect(fault_disarm() == 1u && status == AUTOTUNE_RETRY_EXIT &&
		       occurrences(text, "storage: load point ") == 0 &&
		       waited <= LEASE_SECONDS + COOL_POLL_SECONDS,
	       "a cool wait the drives never end stops at the lease's end, a poll after it at most, its point "
	       "unmeasured");
#endif
	const uint32_t second_cut = POINTS_SECOND_RUN - stopped_measured;
	(void)snprintf(cut, sizeof(cut), "--points_per_run=%u", second_cut);
	char *second_run[] = { program,	       command, source, stage,
			       profile_option, window,	cut };
	status = good ? autotune_captured((int)(sizeof(second_run) /
						sizeof(second_run[0])),
					  second_run, capture, text,
					  sizeof(text)) :
			-1;
	uint32_t resumed = occurrences(text, "storage: load point ");
	(void)snprintf(abba_point, sizeof(abba_point),
		       "storage: load point %u/",
		       POINTS_FIRST_RUN + stopped_measured + 1u);
	expect(strstr(text, "storage: load point ") == strstr(text, abba_point),
	       "the next run measures first the point a stop left unmeasured");
	(void)snprintf(abba_point, sizeof(abba_point),
		       "storage: load point %u/", LOAD_FIRST_ABBA_POINT);
	const char *warmed = strstr(
		text, "storage: load warm-up before the A D D A D A A D: ");
	const char *abba_begins = strstr(text, abba_point);
	grid_warmed = strstr(text, "storage: load warm-up before the grid: ");
	expect(status == AUTOTUNE_RETRY_EXIT && resumed == second_cut &&
		       occurrences(text, "storage: load warm-up ") ==
			       WARMUPS_RESUMED_IN_GRID &&
		       grid_warmed != NULL && grid_warmed < warmed &&
		       warmed != NULL && abba_begins != NULL &&
		       warmed < abba_begins,
	       "a run resumed inside the grid warms up before its first grid point, and again before the "
	       "confirmation's first point");
	char *third_run[] = { program, command,	       source,
			      stage,   profile_option, window };
	status = good ? autotune_captured(
				(int)(sizeof(third_run) / sizeof(third_run[0])),
				third_run, capture, text, sizeof(text)) :
			-1;
	resumed += occurrences(text, "storage: load point ");
	(void)snprintf(abba_point, sizeof(abba_point),
		       "storage: load point %u/", LOAD_FIRST_ABBA_POINT + 1u);
	warmed = strstr(text,
			"storage: load warm-up before the A D D A D A A D: ");
	abba_begins = strstr(text, abba_point);
	expect(occurrences(text, "storage: load warm-up ") == 1u &&
		       warmed != NULL && abba_begins != NULL &&
		       warmed < abba_begins,
	       "a run resumed inside the A D D A warms up again before the first point it measures");
	const bool profiled = file_text(profile, written, sizeof(written));
	const uint32_t grid = occurrences(written, "\"phase\":\"grid\"");
	const uint32_t abba = occurrences(written, "\"phase\":\"abba\"");
	expect(status == EXIT_SUCCESS && profiled &&
		       strstr(written, "\"warmup\":{\"bytes\":") != NULL &&
		       strstr(written, "\"grid_warmup\":{\"bytes\":") != NULL &&
		       strstr(written, "\"spread_over_bound\":") != NULL &&
		       abba == AUTOTUNE_LOAD_ABBA_POINTS &&
		       grid == AUTOTUNE_LOAD_GRID_POINTS &&
		       occurrences(written, "\"pass\":2,") ==
			       AUTOTUNE_LOAD_GRID_CELLS &&
		       occurrences(written, "\"vmstat\":{") == grid + abba &&
		       occurrences(written, "\"cool_wait_nanoseconds\":") ==
			       grid + abba &&
		       resumed + POINTS_FIRST_RUN + stopped_measured ==
			       grid + abba &&
		       access(state, F_OK) != 0,
	       "the next run resumes with the points left, writes every point (its pass, CPU split, page counters and "
	       "cool wait) and removes its progress");
	/* SFF6: the warm-up's passes touch exactly the A D D A point's extents, whose whole size is not
	 * 2 x --window_gib */
	const uint64_t warmup_bytes =
		number_after(written, "\"warmup\":{\"bytes\":");
	bool same_bytes =
		warmup_bytes != 0 && warmup_bytes != UINT64_MAX &&
		warmup_bytes !=
			ABBA_WINDOW_FACTOR * (uint64_t)STAGE_WINDOW_GIB * GIB &&
		number_after(strstr(written, "\"warmup\":{"), "\"passes\":") ==
			LOAD_WARMUP_PASSES_EXPECTED;
	const char *abba_entry = strstr(written, "\"phase\":\"abba\"");
	for (uint32_t point = 0; point < AUTOTUNE_LOAD_ABBA_POINTS; point++) {
		same_bytes = same_bytes && abba_entry != NULL &&
			     number_after(abba_entry, ",\"bytes\":") ==
				     warmup_bytes;
		abba_entry = abba_entry != NULL ? strstr(abba_entry + 1,
							 "\"phase\":\"abba\"") :
						  NULL;
	}
	expect(same_bytes,
	       "the warm-up's passes touch the A D D A points' bytes (their whole extents), not twice the window");
	expect(number_after(strstr(written, "\"grid_warmup\":{"),
			    "\"passes\":") == LOAD_WARMUP_PASSES_EXPECTED,
	       "the grid's warm-up is recorded beside the points with its passes, never ranked");
	expect(strstr(written, ENGRAM_MEMBER ",\"load\":{") != NULL &&
		       strstr(written,
			      "\"enabled\":0,\"qualification\":\"candidate\"") !=
			       NULL &&
		       occurrences(written, "\"key\":") == STAGE_COPIES &&
		       strstr(written, "\"selection\":{\"winner\":{") != NULL &&
		       strstr(written, "\"millicelsius_before\":{") != NULL,
	       "the section is a candidate naming both copies, with the points' temperatures and the selection, "
	       "after the other member kept byte for byte");
	struct storage_load_section section;
	char reason[STORAGE_PROFILE_REASON_BYTES];
	const char *const copies[] = { source, replica };
	const char *const swapped[] = { replica, source };
	const bool read = storage_load_section_read(profile, &section, reason);
	expect(read && section.copy_count == STAGE_COPIES &&
		       storage_load_section_matches(
			       &section, &header, copies, STAGE_COPIES, false,
			       TP_LOAD_HOST_THREADS_DEFAULT,
			       TP_LOAD_SPLIT_STEAL, reason),
	       "the section matches this pack and these copies");
	expect(read &&
		       !storage_load_section_matches(
			       &section, &header, copies, 1u, false,
			       TP_LOAD_HOST_THREADS_DEFAULT,
			       TP_LOAD_SPLIT_STEAL, reason) &&
		       strcmp(reason, "load_section_stale_copies") == 0 &&
		       !storage_load_section_matches(
			       &section, &header, swapped, STAGE_COPIES, false,
			       TP_LOAD_HOST_THREADS_DEFAULT,
			       TP_LOAD_SPLIT_STEAL, reason) &&
		       strcmp(reason, "load_section_stale_copies") == 0,
	       "the source alone, or the copies in another order, are not the section's copies");
	char *qualify_run[] = { program, command,	 source,
				stage,	 profile_option, qualify };
	status = good ? autotune_captured((int)(sizeof(qualify_run) /
						sizeof(qualify_run[0])),
					  qualify_run, capture, text,
					  sizeof(text)) :
			-1;
	expect(status == EXIT_SUCCESS &&
		       storage_load_section_read(profile, &section, reason) &&
		       section.enabled == 1u,
	       "--qualify enables the section that names this pack and these copies");
	(void)unlink(receipt);
	(void)unlink(replica);
	(void)rmdir(replicas);
	(void)unlink(profile);
	(void)unlink(state);
	(void)rmdir(profiles);
	(void)unlink(source);
	if (saved_home[0] != '\0')
		(void)setenv("HOME", saved_home, 1);
	engram_close(engram);
	free(pack);
}

int main(void)
{
	/* every check's line reaches the log even when a sanitizer ends the process at exit (unflushed: batch ac3) */
	(void)setvbuf(stdout, NULL, _IOLBF, 0);
	verdict_cases();
	order_cases();
	rank_cases();
	cool_cases();
	choice_cases();
	const char *const temporary = getenv("TMPDIR");
	char folder[PATH_MAX];
	const int length = snprintf(folder, sizeof(folder),
				    "%s/anemone-load-stage-XXXXXX",
				    temporary != NULL ? temporary : "/tmp");
	if (length <= 0 || (size_t)length >= sizeof(folder) ||
	    mkdtemp(folder) == NULL) {
		(void)puts("FAILED: no temporary folder");
		return EXIT_FAILURE;
	}
	option_cases(folder);
	section_cases(folder);
	early_refusal_cases(folder);
	const char *const first = getenv("ANEMONE_NATIVE_COPY_TEST_FIRST");
	const char *const second = getenv("ANEMONE_NATIVE_COPY_TEST_SECOND");
	char case_folder[PATH_MAX];
	const int case_length =
		first != NULL ? snprintf(case_folder, sizeof(case_folder),
					 "%s/load-case-XXXXXX", first) :
				-1;
	if (first != NULL && second != NULL && case_length > 0 &&
	    (size_t)case_length < sizeof(case_folder) &&
	    mkdtemp(case_folder) != NULL) {
		stage_case(case_folder, second);
		exl3_key_case(case_folder);
		exl3_plan_case(case_folder);
		expect(rmdir(case_folder) == 0,
		       "the stage case leaves its scratch folder empty");
	} else if (first != NULL && second != NULL)
		expect(false, "the stage case's scratch folder is made");
	else
		(void)fputs(
			"storage_autotune_load: the end-to-end case requires the two-device scratch environment\n",
			stderr);
	expect(rmdir(folder) == 0, "the cases leave their folder empty");
	(void)printf("storage_autotune_load_test: %s\n",
		     failures == 0 ? "passed" : "FAILED");
	return failures != 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
