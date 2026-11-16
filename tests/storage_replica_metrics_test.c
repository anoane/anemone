/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/storage_profile.h"
#include <stdlib.h>
#include <string.h>

#define OBSERVATIONS 2u
#define POSITIVE_WEIGHT 3u
#define MOST_SAMPLE_LINES (STORAGE_PROFILE_OBSERVATIONS * 2u)

static bool canonical_case(void)
{
	static struct storage_profile profile;
	profile.load_enabled = 1;
	profile.engram_enabled = 1;
	profile.observation_count = OBSERVATIONS;
	profile.observations[0] = (struct storage_replica_observation){
		.mount = "/mnt/test",
		.reason = "verified_identity_and_bytes",
		.included = true,
		.load_active = true,
		.lookup_active = true,
		.load_weight = POSITIVE_WEIGHT,
		.lookup_weight = 1,
		.physical = { 1 },
		.physical_count = 1
	};
	profile.observations[1] = (struct storage_replica_observation){
		.mount = "/mnt/alias",
		.reason = "duplicate_physical_device",
		.physical = { 1 },
		.physical_count = 1
	};
	char *text = NULL;
	size_t bytes = 0;
	FILE *stream = open_memstream(&text, &bytes);
	if (stream == NULL)
		return false;
	if (storage_profile_metrics(stream, &profile, false) != STATUS_OK ||
	    fclose(stream) != 0 || bytes != 0) {
		free(text);
		return false;
	}
	free(text);
	text = NULL;
	stream = open_memstream(&text, &bytes);
	if (stream == NULL)
		return false;
	const result_status status =
		storage_profile_metrics(stream, &profile, true);
	const int closed = fclose(stream);
	static const char expected[] =
		"# HELP anemone_storage_replica_info Startup replica selection by physical device and role.\n"
		"# TYPE anemone_storage_replica_info gauge\n"
		"anemone_storage_replica_info{mount=\"/mnt/test\",role=\"load\",status=\"included\","
		"reason=\"verified_identity_and_bytes\"} 1\n"
		"anemone_storage_replica_info{mount=\"/mnt/test\",role=\"lookup\",status=\"included\","
		"reason=\"verified_identity_and_bytes\"} 1\n";
	const bool passed = status == STATUS_OK && closed == 0 &&
			    strcmp(text, expected) == 0;
	free(text);
	return passed;
}

/* All-RAM Engram (no disk lookup): an included copy's lookup role is a disabled action, not a zero tuned weight. */
static bool disabled_lookup_case(void)
{
	static struct storage_profile profile;
	profile.load_enabled = 1;
	profile.engram_enabled = 0;
	profile.observation_count = 1;
	profile.observations[0] =
		(struct storage_replica_observation){ .mount = "/mnt/test",
						      .reason = "eligible",
						      .included = true,
						      .load_active = true,
						      .physical = { 1 },
						      .physical_count = 1 };
	char *text = NULL;
	size_t bytes = 0;
	FILE *const stream = open_memstream(&text, &bytes);
	if (stream == NULL)
		return false;
	const result_status status =
		storage_profile_metrics(stream, &profile, true);
	const int closed = fclose(stream);
	const bool passed =
		status == STATUS_OK && closed == 0 &&
		strstr(text,
		       "anemone_storage_replica_info{mount=\"/mnt/test\",role=\"lookup\",status=\"excluded\","
		       "reason=\"action_disabled\"} 1\n") != NULL;
	free(text);
	return passed;
}

static bool limits_case(void)
{
	static struct storage_profile profile;
	profile.observation_count = STORAGE_PROFILE_OBSERVATIONS;
	for (uint32_t index = 0; index < STORAGE_PROFILE_OBSERVATIONS;
	     index++) {
		struct storage_replica_observation *const entry =
			&profile.observations[index];
		(void)snprintf(entry->mount, sizeof(entry->mount),
			       "/mnt/%u\\\"\n", index);
		(void)snprintf(entry->reason, sizeof(entry->reason), "%s",
			       "device_disappeared_or_unknown");
		entry->physical[0] = index + 1u;
		entry->physical_count = 1;
	}
	char *text = NULL;
	size_t bytes = 0;
	FILE *const stream = open_memstream(&text, &bytes);
	if (stream == NULL)
		return false;
	const result_status status =
		storage_profile_metrics(stream, &profile, true);
	const int closed = fclose(stream);
	uint32_t samples = 0;
	if (status == STATUS_OK && closed == 0)
		for (const char *cursor = text;
		     (cursor = strstr(cursor,
				      "anemone_storage_replica_info{mount=")) !=
		     NULL;
		     cursor++)
			samples++;
	const bool bounded = samples == MOST_SAMPLE_LINES &&
			     bytes < STORAGE_PROFILE_METRICS_BYTES;
	const bool escaped =
		text != NULL &&
		strstr(text, "mount=\"/mnt/0\\\\\\\"\\n\",role=\"load\"") !=
			NULL;
	const bool failed_visible =
		text != NULL &&
		strstr(text,
		       "status=\"excluded\",reason=\"device_disappeared_or_unknown\"") !=
			NULL;
	free(text);
	profile.observation_count = STORAGE_PROFILE_OBSERVATIONS + 1u;
	return bounded && escaped && failed_visible &&
	       storage_profile_metrics(stdout, &profile, true) ==
		       STATUS_INVALID_ARGUMENT;
}

int main(void)
{
	const bool disabled = disabled_lookup_case();
	if (!disabled)
		(void)fputs(
			"FAILED: all-RAM lookup role reports action_disabled\n",
			stderr);
	return canonical_case() && limits_case() && disabled ? EXIT_SUCCESS :
							       EXIT_FAILURE;
}
