/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef STORAGE_AUTOTUNE_H
#define STORAGE_AUTOTUNE_H
#include "engine/storage/replicas/replica_prepare.h"
#include "engine/engram/engram.h"
#include "engine/engram/engram_disk.h"
#include "engine/storage/profile/storage_profile.h"
/* What every stage's command line shares: the options begin after `anemone autotune`; a lease of at most
 * AUTOTUNE_SECONDS_MOST seconds (default AUTOTUNE_SECONDS_DEFAULT) that ends unfinished exits AUTOTUNE_RETRY_EXIT, and
 * the same command resumes it. */
#define AUTOTUNE_FIRST_OPTION 2
#define AUTOTUNE_SECONDS_DEFAULT 1650u
#define AUTOTUNE_SECONDS_MOST 1700u
#define AUTOTUNE_RETRY_EXIT 75
#define AUTOTUNE_STAGE_OPTION "--stage="
/* The monotonic clock and the process's CPU time (user and system), in nanoseconds. */
uint64_t autotune_now_nanoseconds(void);
uint64_t autotune_cpu_nanoseconds(void);
/* Device telemetry both tuning stages record beside their points (on the device-discovery side of the owner's split,
 * 9 Nov 2026, as the clocks above are plumbing): every NVMe controller's composite temperature (its first hwmon's
 * temp1_input), at most AUTOTUNE_DRIVES_MOST; none when sysfs has none. autotune_drives_json writes
 * ,"NAME":{"nvme0":millicelsius,...}. */
#define AUTOTUNE_DRIVES_MOST 16u
#define AUTOTUNE_DRIVE_NAME_BYTES 32u
struct autotune_drives {
	uint32_t count;
	char names[AUTOTUNE_DRIVES_MOST][AUTOTUNE_DRIVE_NAME_BYTES];
	int64_t millicelsius[AUTOTUNE_DRIVES_MOST];
};
void autotune_drives_read(struct autotune_drives *drives);
void autotune_drives_json(FILE *output, const char *name,
			  const struct autotune_drives *drives);
/* A decimal of at most most; false for anything else. */
bool autotune_number_option(const char *text, uint32_t most, uint32_t *output);
/* Every refusal names its reason and path on stderr; false, for a validity chain. */
bool autotune_refused(const char *reason, const char *path);
/* The profile's default path: $XDG_CONFIG_HOME/anemone/storage-profile.json, else $HOME/.config/anemone/...; false,
 * the refusal named (autotune_refused), when neither is set or the path does not fit. */
bool autotune_profile_default(char *output, size_t capacity);
/* The copy stage's decisions beside the profile (<profile>.copy.json): whether the copy completed, its parameters
 * (copy_workers) and one decision a device. */
MUST_CHECK result_status
prepare_decisions_write(const char *path, const struct prepare_target *targets,
			uint32_t count, bool complete, uint32_t copy_workers);
void prepare_decision_json(FILE *output, const struct prepare_target *target);
/* The device an explicit destination directory will be on: the directory's description, else, while the path does not
 * exist yet, its nearest existing ancestor's, with the mount entries of mount_table (the product's is
 * /proc/self/mounts). An automount that has not fired fails (STATUS_UNSUPPORTED, device naming autofs). */
MUST_CHECK result_status
prepare_destination_device(const char *mount_table, const char *directory,
			   struct local_device_description *device);
int storage_autotune_cli(int count, char **arguments);
/* The combined tuning's --prompt, retired with it (v0.0.4): every stage refuses it by name, with this reason. */
#define AUTOTUNE_PROMPT_RETIRED \
	"--prompt went with the combined tuning: the Engram stage replays every --tokens case whole"
/* anemone autotune PACK --stage=load (storage_autotune_load.c): the load stage. */
int storage_autotune_load_cli(int count, char **arguments);
/* Whether a stage's section of the profile is current: it reads and names this pack (source, a real path) and exactly
 * the copies the stage would measure now, by their device keys in order (the load stage: the copies serving finds,
 * in its host shape; the Engram stage: the copies serving's disk store admits, in serving's flight shape), a candidate
 * or qualified. False with reason otherwise: the section's own reason (missing, no_load_section,
 * engram_section_stale_copies, ...), or AUTOTUNE_COPIES_UNOPENED. No copy is opened without a section.
 * `anemone autotune PACK` without --stage skips a stage whose section is current; with --retune only one that finished
 * earlier in the same --retune series (<profile>.retune-series names each stage that finished since the series' first
 * run; a run that ends with an exit other than AUTOTUNE_RETRY_EXIT ends the series). */
#define AUTOTUNE_COPIES_UNOPENED "copies_unopened"
bool storage_autotune_load_current(const char *source, const char *profile,
				   bool external,
				   char reason[STORAGE_PROFILE_REASON_BYTES]);
bool storage_autotune_engram_current(const char *source, const char *profile,
				     bool external,
				     char reason[STORAGE_PROFILE_REASON_BYTES]);
/* The load stage's grid (17 Nov 2026): its AUTOTUNE_LOAD_GRID_CELLS arms (readers a copy x piece) run in a fixed
 * balanced order, then once more in the reverse order, AUTOTUNE_LOAD_GRID_POINTS points; autotune_load_arm_of gives
 * grid point index's arm. autotune_load_grid_rank ranks every arm by the mean of its two points' rates (GB/s, in grid
 * point order) and returns the challenger: the fastest arm other than serving's default (*fallback), a mean equal to
 * another's going to the simpler arm; each arm by the index of its point in the first pass. */
#define AUTOTUNE_LOAD_GRID_CELLS \
	35u /* readers 0, 1, 2, 3, 4, 6, 8 x pieces 4 to 64 MiB */
#define AUTOTUNE_LOAD_GRID_PASSES 2u
#define AUTOTUNE_LOAD_GRID_POINTS \
	(AUTOTUNE_LOAD_GRID_CELLS * AUTOTUNE_LOAD_GRID_PASSES)
void autotune_load_arm_of(uint32_t index, uint32_t *readers,
			  uint32_t *piece_mib);
uint32_t autotune_load_grid_rank(const double rates[AUTOTUNE_LOAD_GRID_POINTS],
				 uint32_t *fallback);
/* The load stage's confirmation verdict over the rates of its eight points in order, A D D A D A A D: the challenger
 * (A), the default (D). The higher mean wins even inside the noise (the owner's rule, 29 Oct 2026); means equal at the
 * reported precision (AUTOTUNE_LOAD_RATE_SCALE: GB/s to three decimals) go to the simpler arm (default_simpler: the
 * default has fewer readers a copy, no cap counting as the most, or as many and a smaller piece). The spread (the
 * larger of the two arms' ranges over their means) and the margin (the challenger's mean over the default's, less one)
 * are recorded beside the choice; a spread over AUTOTUNE_LOAD_SPREAD_BOUND is flagged, never refused: the verdict
 * stands and the points want a look before --qualify (the load stage's first real run, 15 Nov 2026: one cold point gave
 * a spread of 137 %). Both arms sit at the same mean position, so a linear drift cancels (a grid run of 15 Nov 2026: a
 * 4-point spread of 5.70 %). */
#define AUTOTUNE_LOAD_ABBA_POINTS 8u
#define AUTOTUNE_LOAD_RATE_SCALE 1000.0
#define AUTOTUNE_LOAD_SPREAD_BOUND 0.25 /* an arm's range over its mean */
struct autotune_load_verdict {
	double spread, margin;
	bool exact_tie, default_wins, spread_over_bound;
};
void autotune_load_verdict(const double rates[AUTOTUNE_LOAD_ABBA_POINTS],
			   bool default_simpler,
			   struct autotune_load_verdict *verdict);
/* Before each point the load stage waits until every NVMe drive's composite is below AUTOTUNE_LOAD_COOL_MILLICELSIUS
 * (the 9100 PRO warns at 83.85 C), at most AUTOTUNE_LOAD_COOL_SECONDS_MOST and never past the lease; a stop signal or
 * the lease's end during the wait leaves the point unmeasured. autotune_load_drives_cool says whether drives are cool;
 * autotune_load_cool_end gives the latest end of a wait begun at begin (nanoseconds, autotune_now_nanoseconds): the
 * bound later, or the lease's deadline when it comes first. */
#define AUTOTUNE_LOAD_COOL_MILLICELSIUS 60000
#define AUTOTUNE_LOAD_COOL_SECONDS_MOST 180u
bool autotune_load_drives_cool(const struct autotune_drives *drives);
uint64_t autotune_load_cool_end(uint64_t begin, uint64_t deadline);
/* A load point's process counters (the load stage's, and tools/measure/load_arena_probe's): user and system CPU apart
 * (getrusage), and the /proc/vmstat counters of the arena's first touch, autotune_load_vmstat_names, each with its bit
 * in vmstat_read (a counter the kernel lacks is recorded as missing: null). autotune_load_counters_since gives the
 * counters since before; autotune_load_counters_json writes them as ,"user_nanoseconds":...,"vmstat":{...}. */
#define AUTOTUNE_LOAD_VMSTAT_COUNTERS 5u
extern const char *const
	autotune_load_vmstat_names[AUTOTUNE_LOAD_VMSTAT_COUNTERS];
struct autotune_load_counters {
	uint64_t user_nanoseconds, system_nanoseconds;
	uint64_t vmstat[AUTOTUNE_LOAD_VMSTAT_COUNTERS];
	uint32_t vmstat_read;
};
void autotune_load_counters_read(struct autotune_load_counters *counters);
void autotune_load_counters_since(const struct autotune_load_counters *before,
				  struct autotune_load_counters *since);
void autotune_load_counters_json(FILE *output,
				 const struct autotune_load_counters *counters);
/* anemone autotune PACK --stage=engram (engram_autotune.c): the Engram stage, and what its test reaches. */
int storage_autotune_engram_cli(int count, char **arguments);
#define AUTOTUNE_ENGRAM_LEVELS 7u /* D1, D2, D4, D8, P1, P2, M */
#define AUTOTUNE_ENGRAM_ABBA_POINTS 4u
#define AUTOTUNE_ENGRAM_TRACES_MOST 16u /* --tokens files */
/* one lease bound for the Engram stage: its default is its most */
#define AUTOTUNE_ENGRAM_SECONDS_MOST AUTOTUNE_SECONDS_DEFAULT
#define AUTOTUNE_ENGRAM_GRID_POINTS \
	32u /* threads 1-4 x spin 0, 50, 200, 1000 us x pin 0, 1 */
#define AUTOTUNE_ENGRAM_POINTS_MOST \
	(2u * AUTOTUNE_ENGRAM_GRID_POINTS + 2u * AUTOTUNE_ENGRAM_ABBA_POINTS)
#define AUTOTUNE_ENGRAM_WARMUP_SIDES \
	2u /* before grid and re-measured points, before the A D D A */
#define AUTOTUNE_ENGRAM_SPREAD_BOUND \
	0.10 /* an arm's range of its two costs over their mean */
#define AUTOTUNE_ENGRAM_COST_SCALE \
	0.01 /* the reported precision: 0.1 us, in ns */
#define AUTOTUNE_ENGRAM_CORES_SCALE 100.0 /* cores to two decimals */
/* The Engram stage's A D D A verdict over the first table's p50 (ns) of each level at its four points (point-major:
 * [point * AUTOTUNE_ENGRAM_LEVELS + level]), in order: the
 * challenger (the fastest grid arm other than serving's default), the default, the default, the challenger; and each
 * point's cores (CPU over wall). An arm's cost is the geometric mean over the levels of its two points' mean, so each
 * level counts by its relative change (levels range from microseconds to tens of milliseconds). (b) A level whose mean
 * is slower than the default's by more than that level's own spread (the larger of the two arms' ranges over their
 * means) blocks the challenger: blocked_by names the first, AUTOTUNE_ENGRAM_LEVELS for none (per-split-axes: balanced
 * holds no measured loss). (a) Otherwise the lower cost wins even inside the noise (the owner's rule, 29 Oct 2026). (c)
 * Costs equal at the reported precision go to fewer cores, then to the default. The margin (the default's cost over
 * the challenger's, less one) and the spread (the larger arm's range of its two points' costs over their mean) are
 * recorded beside the choice; a spread over AUTOTUNE_ENGRAM_SPREAD_BOUND is flagged, never refused. */
struct autotune_engram_verdict {
	double challenger_cost, default_cost, margin, spread;
	double level_loss[AUTOTUNE_ENGRAM_LEVELS],
		level_spread[AUTOTUNE_ENGRAM_LEVELS];
	uint32_t blocked_by;
	bool exact_tie, default_holds, spread_over_bound;
};
void autotune_engram_verdict(const double first_ns[AUTOTUNE_ENGRAM_ABBA_POINTS *
						   AUTOTUNE_ENGRAM_LEVELS],
			     const double cores[AUTOTUNE_ENGRAM_ABBA_POINTS],
			     struct autotune_engram_verdict *verdict);
/* The trace: real prompts as token ids (--tokens files, little-endian uint32, a case each), their row ids from the
 * pack's hashing constants ([token][table][rows a token], engram_row_ids_entry's order, every case from its own start),
 * the distinct (table, block) pairs they read, and the identity of the files (the SHA-256 of their SHA-256s in
 * order). No token is recorded anywhere else. A refusal names the file at fault: *failed is its index (count when
 * none is, memory). */
struct autotune_engram_trace {
	uint64_t *row_ids;
	uint32_t case_first[AUTOTUNE_ENGRAM_TRACES_MOST + 1u];
	uint32_t cases, tokens, references_per_token;
	uint64_t unique_blocks;
	char identity[STORAGE_PROFILE_KEY_BYTES];
};
MUST_CHECK result_status autotune_engram_trace_open(
	const char *const *paths, uint32_t count,
	const struct engram_record *engram, struct autotune_engram_trace *trace,
	uint32_t *failed);
void autotune_engram_trace_close(struct autotune_engram_trace *trace);
/* A stream's walk: steps of rows_per_step consecutive tokens of a case (a short case, or a case's end, gives fewer),
 * the cases from first_case on and around, each once. */
struct autotune_engram_walk {
	uint32_t first_case, rows_per_step, walked, offset;
};
bool autotune_engram_walk_next(const struct autotune_engram_trace *trace,
			       struct autotune_engram_walk *walk,
			       uint32_t *first_token, uint32_t *rows);
/* The row a point reads for row_id of a table of row_count rows: its block moved by shift_blocks whole blocks around
 * the table, the row's place in its block kept; a row that would land behind the table's last row stays. */
uint64_t autotune_engram_rotated(uint64_t row_id, uint64_t shift_blocks,
				 uint64_t row_count);
/* One arm of the grid, by its index: threads a copy, the completion spin window, pinning. */
struct autotune_engram_arm {
	uint32_t threads_per_copy, spin_us, pin;
};
void autotune_engram_arm_of(uint32_t arm_index,
			    struct autotune_engram_arm *arm);
enum autotune_engram_phase {
	AUTOTUNE_ENGRAM_WARMUP,
	AUTOTUNE_ENGRAM_GRID,
	AUTOTUNE_ENGRAM_REMEASURE,
	AUTOTUNE_ENGRAM_ADDA,
	AUTOTUNE_ENGRAM_PHASES
};
/* Why a point is disturbed: a ring retired, a read failed (another copy, or the waiter's fallback, gave its block), a
 * copy left failed, a drive over the stage's temperature bound. */
enum autotune_engram_disturbance {
	AUTOTUNE_ENGRAM_QUIET,
	AUTOTUNE_ENGRAM_RETIREMENT,
	AUTOTUNE_ENGRAM_FAILED_READ,
	AUTOTUNE_ENGRAM_FAILED_COPY,
	AUTOTUNE_ENGRAM_HOT_DRIVE,
	AUTOTUNE_ENGRAM_DISTURBANCES
};
struct autotune_engram_level {
	uint64_t steps, rows, wall_ns, first_p50_ns, first_p99_ns,
		first_p999_ns, all_p50_ns, all_p99_ns;
};
struct autotune_engram_point {
	struct autotune_engram_arm arm;
	uint32_t phase, arm_index, rotation, disturbance, retirements,
		plain_rings;
	uint64_t cpu_ns, wall_ns, answered, failed_reads;
	uint64_t copy_reads[STORAGE_MAXIMUM_COPIES];
	struct autotune_engram_level levels[AUTOTUNE_ENGRAM_LEVELS];
	struct autotune_drives before, after;
};
/* One point (point->arm set): a flight of serving's shape over the store's copies with the arm's parameters, its I/O
 * threads pinned where storage_flight_queue_cpu says when the arm pins, the trace at rotation, every level, and with
 * check the identity check, which refuses a difference (STATUS_CORRUPT_PACK, one line naming the point, level, table
 * and step). Every copy starts the point unfailed. */
MUST_CHECK result_status autotune_engram_point_measure(
	struct engram_disk *disk, const struct autotune_engram_trace *trace,
	uint32_t rotation, bool check, struct autotune_engram_point *point);
/* The stage's resumable progress: its identity (the pack, the copies' keys in order, the trace, the flight shape, the
 * order's seed), the last warm-up on each side (before grid and re-measured points, before the A D D A), every
 * point. */
struct autotune_engram_state {
	uint32_t schema, copy_count, steps, slots, point_count, warmup_count;
	char pack_identity[STORAGE_PROFILE_KEY_BYTES],
		trace_identity[STORAGE_PROFILE_KEY_BYTES];
	char keys[STORAGE_MAXIMUM_COPIES][STORAGE_PROFILE_KEY_BYTES];
	uint64_t file_byte_count, seed;
	struct autotune_engram_point warmups[AUTOTUNE_ENGRAM_WARMUP_SIDES];
	struct autotune_engram_point points[AUTOTUNE_ENGRAM_POINTS_MOST];
};
/* The next point from the points done: the grid's arms in the seeded order, then each disturbed grid point's arm once
 * more, then the A D D A of the fastest arm with an undisturbed point other than the default against the default, a
 * slot at a time: a disturbed A D D A point is measured again in its slot, the order kept, and a slot disturbed twice
 * ends the A D D A (the default holds: autotune_engram_select). False when none is left, or with *refusal when the
 * default or every other arm has no undisturbed point. */
bool autotune_engram_next(const struct autotune_engram_state *state,
			  uint32_t *arm_index, uint32_t *phase,
			  const char **refusal);
/* The selection once the schedule is done: the challenger, the grid's and its re-measured points (measured), the point
 * holding each A D D A slot in order (slots, held of them) and the verdict over them (autotune_engram_verdict). When a
 * slot was disturbed twice (held below AUTOTUNE_ENGRAM_ABBA_POINTS, adda_disturbed_twice) the default holds, its reason
 * named, and the verdict's costs are not computed (zero). False with *refusal when no arm ranks, false alone while
 * points remain. */
struct autotune_engram_selection {
	struct autotune_engram_verdict verdict;
	uint32_t challenger, measured, held;
	uint32_t slots[AUTOTUNE_ENGRAM_ABBA_POINTS];
	bool adda_disturbed_twice;
};
bool autotune_engram_select(const struct autotune_engram_state *state,
			    struct autotune_engram_selection *selection,
			    const char **refusal);
/* The prior progress at path into state when it is this user's, whole, of the same identity and in the schedule's
 * order; else state is left as it was. */
void autotune_engram_state_resume(const char *path,
				  struct autotune_engram_state *state);
void storage_autotune_hint(FILE *output);
#endif
