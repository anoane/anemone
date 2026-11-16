/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef STORAGE_AUTOTUNE_H
#define STORAGE_AUTOTUNE_H
#include "replica_prepare.h"
#define PREPARE_TRACE_CASES 16u
/* What every stage's command line shares: the options begin after `anemone autotune`; a lease of at most
 * AUTOTUNE_SECONDS_MOST seconds (default AUTOTUNE_SECONDS_DEFAULT) that ends unfinished exits AUTOTUNE_RETRY_EXIT, and
 * the same command resumes it. */
#define AUTOTUNE_FIRST_OPTION 2
#define AUTOTUNE_SECONDS_DEFAULT 1650u
#define AUTOTUNE_SECONDS_MOST 1700u
#define AUTOTUNE_RETRY_EXIT 75
#define AUTOTUNE_STAGE_OPTION "--stage="
/* A decimal of at most most; false for anything else. */
bool autotune_number_option(const char *text, uint32_t most, uint32_t *output);
/* Every refusal names its reason and path on stderr; false, for a validity chain. */
bool autotune_refused(const char *reason, const char *path);
/* The profile's default path: $XDG_CONFIG_HOME/anemone/storage-profile.json, else $HOME/.config/anemone/...; false
 * when neither is set or the path does not fit. */
bool autotune_profile_default(char *output, size_t capacity);
struct prepare_trace_input {
	const char *path;
	uint32_t prompt;
};
struct prepare_tune_options {
	const char *profile, *manifest_path;
	struct prepare_manifest *manifest;
	struct prepare_target *devices;
	uint32_t count, floor_percent;
	bool external;
	struct prepare_trace_input traces[PREPARE_TRACE_CASES];
	uint32_t trace_count;
	uint64_t deadline_ns;
	const volatile sig_atomic_t *stop;
};
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
MUST_CHECK result_status
prepare_tune(const struct prepare_tune_options *options);
int storage_autotune_cli(int count, char **arguments);
/* anemone autotune PACK --stage=load (storage_autotune_load.c): the load stage. */
int storage_autotune_load_cli(int count, char **arguments);
/* The load stage's A D D A verdict over the rates of its four points in order: the challenger (the fastest grid arm
 * other than serving's default), the default, the default, the challenger. The higher mean wins even inside the noise
 * (the owner's rule, 29 Oct 2026); means equal at the reported precision (AUTOTUNE_LOAD_RATE_SCALE: GB/s to three
 * decimals) go to the simpler arm (default_simpler: the default has fewer readers a copy, no cap counting as the most,
 * or as many and a smaller piece). The spread (the larger of the two arms' ranges over their means) and the margin
 * (the challenger's mean over the default's, less one) are recorded beside the choice; a spread over
 * AUTOTUNE_LOAD_SPREAD_BOUND is flagged, never refused: the verdict stands and the points want a look before
 * --qualify (R26, 15 Nov 2026: one cold point gave a spread of 137 %). */
#define AUTOTUNE_LOAD_ABBA_POINTS 4u
#define AUTOTUNE_LOAD_RATE_SCALE 1000.0
#define AUTOTUNE_LOAD_SPREAD_BOUND 0.25 /* an arm's range over its mean */
struct autotune_load_verdict {
	double spread, margin;
	bool exact_tie, default_wins, spread_over_bound;
};
void autotune_load_verdict(const double rates[AUTOTUNE_LOAD_ABBA_POINTS],
			   bool default_simpler,
			   struct autotune_load_verdict *verdict);
void storage_autotune_hint(FILE *output);
#endif
