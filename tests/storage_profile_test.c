/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/storage_profile.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SENTINEL 73u
#define LOAD_SHARE 3u
#define LOOKUP_ONLY_COPIES 2u
static unsigned failures;
static void expect(bool good, const char *message)
{
	if (!good) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", message);
	}
}

static void refusal_cases(void)
{
	struct storage_profile profile = { .count = SENTINEL };
	char reason[STORAGE_PROFILE_REASON_BYTES];
	char key[STORAGE_PROFILE_KEY_BYTES] = "unchanged";
	uint64_t physical = SENTINEL;
	expect(!storage_profile_read(NULL, &profile, reason) &&
		       profile.count == SENTINEL &&
		       !storage_profile_read("/proc/version", &profile,
					     reason) &&
		       profile.count == SENTINEL &&
		       strcmp(reason, "unreadable_or_invalid_json") == 0 &&
		       storage_device_key(NULL, key, &physical) ==
			       STATUS_INVALID_ARGUMENT &&
		       storage_device_key("/proc/version", key, &physical) ==
			       STATUS_UNSUPPORTED &&
		       strcmp(key, "unchanged") == 0 && physical == SENTINEL,
	       "invalid profiles and keys leave outputs untouched");
}

/* A zero load share keeps a device for lookup only: never in the load list (the loader refuses zero shares). */
static void load_list_cases(void)
{
	static struct storage_profile profile;
	profile.count = LOOKUP_ONLY_COPIES;
	(void)snprintf(profile.paths[0], sizeof(profile.paths[0]), "%s",
		       "/mnt/a/model.anemone");
	(void)snprintf(profile.paths[1], sizeof(profile.paths[1]), "%s",
		       "/mnt/b/model.anemone");
	profile.load_weights[0] = LOAD_SHARE;
	profile.engram_weights[1] = LOAD_SHARE;
	const char *paths[STORAGE_MAXIMUM_COPIES];
	uint32_t weights[STORAGE_MAXIMUM_COPIES];
	expect(storage_profile_load_list(&profile, paths, weights) == 1u &&
		       paths[0] == profile.paths[0] && weights[0] == LOAD_SHARE,
	       "a zero load share is a lookup-only device, absent from the load list");
}

/* The disk lookup sources are the verified packs: an EXL3 map (an EXL3 profile's source) is left out, since the disk
 * store cannot open it; a map alone leaves none. A list that does not fit is refused, and empty. */
static void engram_source_cases(void)
{
	static struct storage_profile profile;
	static const char *const devices[] = { "/mnt/a/model.exl3map",
					       "/mnt/b/flat.anemone",
					       "/mnt/c/flat.anemone" };
	const uint32_t device_count = sizeof(devices) / sizeof(devices[0]);
	profile.count = device_count;
	for (uint32_t index = 0; index < device_count; index++)
		(void)snprintf(profile.paths[index],
			       sizeof(profile.paths[index]), "%s",
			       devices[index]);
	char sources[sizeof(profile.paths)];
	uint32_t count = SENTINEL;
	expect(storage_profile_engram_sources(&profile, sources,
					      sizeof(sources), &count) &&
		       count == profile.count - 1u &&
		       strcmp(sources,
			      "/mnt/b/flat.anemone,/mnt/c/flat.anemone") == 0,
	       "an EXL3 map is no disk lookup source; its flat copies are");
	count = SENTINEL;
	expect(!storage_profile_engram_sources(&profile, sources,
					       strlen(devices[1]), &count) &&
		       count == SENTINEL && sources[0] == '\0',
	       "a list that does not fit is refused and left empty");
	profile.count = 1;
	expect(storage_profile_engram_sources(&profile, sources,
					      sizeof(sources), &count) &&
		       count == 0 && sources[0] == '\0',
	       "a profile whose only verified device is an EXL3 map has no disk lookup source");
}

static void engram_family_cases(void)
{
	expect(storage_profile_engram_argument("--engram_io=1") &&
		       storage_profile_engram_argument(
			       "--engram_ram_percent=50") &&
		       storage_profile_engram_argument(
			       "--engram_cache_mib=256") &&
		       storage_profile_engram_argument(
			       "--engram_sources=/a,/b") &&
		       storage_profile_engram_argument(
			       "--turbo-nvme-engrams=1"),
	       "the disk lookup family overrides its profile");
	expect(!storage_profile_engram_argument("--engram_shared=1") &&
		       !storage_profile_engram_argument("--turbo_load=1"),
	       "--engram_shared (TP staging) keeps the tuned lookup profile");
}

/* Diagnostics never fail startup: a full observation table drops a new mount's entry. */
static void observation_cases(void)
{
	char reason[STORAGE_PROFILE_REASON_BYTES];
	struct storage_profile unused;
	(void)storage_profile_read("/nonexistent/anemone-profile.json", &unused,
				   reason); /* resets the decisions */
	bool noted = true;
	for (uint32_t index = 0; index <= STORAGE_PROFILE_OBSERVATIONS;
	     index++) {
		struct local_device_description device = {
			.count = 1, .physical = { index + 1u }
		};
		(void)snprintf(device.mount, sizeof(device.mount),
			       "/mnt/full%u", index);
		noted = noted &&
			storage_profile_note(&device,
					     "device_disappeared_or_unknown",
					     STORAGE_REPLICA_LOOKUP) ==
				STATUS_OK;
	}
	expect(noted && storage_profile_decisions()->observation_count ==
				STORAGE_PROFILE_OBSERVATIONS,
	       "a full observation table drops the 65th entry and startup continues");
}

/* An untuned startup loads its given copy whatever the media: reported included, the refusal advisory. */
static void loaded_copy_cases(void)
{
	char reason[STORAGE_PROFILE_REASON_BYTES];
	struct storage_profile unused;
	(void)storage_profile_read("/nonexistent/anemone-profile.json", &unused,
				   reason);
	struct local_device_description device = { .count = 1,
						   .physical = { SENTINEL },
						   .rotational = true,
						   .mount = "/mnt/disk",
						   .model = "fixture_hdd",
						   .transport = "internal" };
	char *text = NULL;
	size_t bytes = 0;
	FILE *const stream = open_memstream(&text, &bytes);
	expect(stream != NULL, "loaded-copy capture opens");
	if (stream == NULL)
		return;
	const result_status noted = storage_profile_note_loaded(
		stream, &device, "rotational_device",
		LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT);
	const bool closed = fclose(stream) == 0;
	const struct storage_profile *const decisions =
		storage_profile_decisions();
	expect(noted == STATUS_OK && closed &&
		       strstr(text, "included: eligible") != NULL &&
		       strstr(text, "excluded") == NULL &&
		       strstr(text, "advisory: rotational_device") != NULL &&
		       decisions->observation_count == 1u &&
		       decisions->observations[0].included &&
		       decisions->observations[0].load_active,
	       "the copy being loaded is reported included, its media refusal advisory");
	free(text);
}

/* Untuned with disk lookup: the loaded copy's lookup role refused for its media reports that reason, not
 * zero_tuned_weight. */
static void refused_role_cases(void)
{
	char reason[STORAGE_PROFILE_REASON_BYTES];
	struct storage_profile unused;
	(void)storage_profile_read("/nonexistent/anemone-profile.json", &unused,
				   reason);
	storage_profile_roles(false, false, true);
	struct local_device_description device = { .count = 1,
						   .physical = { SENTINEL },
						   .rotational = true,
						   .mount = "/mnt/bulk",
						   .model = "fixture_hdd",
						   .transport = "internal" };
	char *text = NULL;
	size_t bytes = 0;
	FILE *const lines = open_memstream(&text, &bytes);
	expect(lines != NULL, "refused-role capture opens");
	if (lines == NULL)
		return;
	const bool noted =
		storage_profile_note_loaded(
			lines, &device, "rotational_device",
			LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT) == STATUS_OK &&
		storage_profile_note(&device, "rotational_device",
				     STORAGE_REPLICA_LOOKUP) == STATUS_OK;
	const bool metrics =
		storage_profile_metrics(lines, storage_profile_decisions(),
					true) == STATUS_OK;
	const bool closed = fclose(lines) == 0;
	expect(closed && text != NULL && noted && metrics &&
		       strstr(text,
			      "mount=\"/mnt/bulk\",role=\"load\","
			      "status=\"included\",reason=\"eligible\"") !=
			       NULL &&
		       strstr(text, "mount=\"/mnt/bulk\",role=\"lookup\","
				    "status=\"excluded\","
				    "reason=\"rotational_device\"") != NULL,
	       "a refused lookup role of the loaded copy reports its refusal reason");
	free(text);
}

int main(void)
{
	refusal_cases();
	load_list_cases();
	engram_source_cases();
	engram_family_cases();
	observation_cases();
	loaded_copy_cases();
	refused_role_cases();
	return failures != 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
