/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef STORAGE_PROFILE_H
#define STORAGE_PROFILE_H
#include <stdbool.h>
#include <stdio.h>
#include "engine/storage/storage_seam.h"
#include "engine/storage/replicas/replicas.h"
#include "engine/storage/io/local_device.h"
#define STORAGE_PROFILE_KEY_BYTES 65u
#define STORAGE_PROFILE_REASON_BYTES 128u
#define STORAGE_PROFILE_OBSERVATIONS 64u
struct storage_replica_observation {
	char mount[REPLICA_PATH_CAPACITY], reason[STORAGE_PROFILE_REASON_BYTES];
	/* why one role of an included copy was refused (empty when it was not): media or duplicate, say */
	char load_reason[STORAGE_PROFILE_REASON_BYTES],
		lookup_reason[STORAGE_PROFILE_REASON_BYTES];
	uint64_t physical[LOCAL_DEVICE_PHYSICAL_MOST];
	uint32_t physical_count, load_weight, lookup_weight;
	bool included, load_active, lookup_active;
};
struct storage_profile {
	char identity[STORAGE_PROFILE_KEY_BYTES];
	char source[REPLICA_PATH_CAPACITY];
	char pack_identity[STORAGE_PROFILE_KEY_BYTES];
	uint64_t source_bytes;
	bool source_unavailable;
	struct storage_replica_observation
		observations[STORAGE_PROFILE_OBSERVATIONS];
	uint32_t observation_count;
	char paths[STORAGE_MAXIMUM_COPIES][REPLICA_PATH_CAPACITY];
	uint32_t load_weights[STORAGE_MAXIMUM_COPIES];
	uint32_t engram_weights[STORAGE_MAXIMUM_COPIES];
	uint32_t count, excluded, load_enabled, engram_enabled;
	uint32_t replica_external, replica_floor_percent;
	uint32_t load_threads, ram_percent, cache_mib, engram_io;
	uint64_t created_unix;
};
/* The load stage's section of a schema 3 profile (anemone autotune PACK --stage=load; docs/user/STORAGE_PROFILE.md).
 * A schema 3 profile's top level holds its schema, the source's real path and one member a stage; a stage rewrites
 * only its own member, and every other byte stays. storage_load_section_read reads the section's identity (the pack's
 * header digest and size, the copies' device keys in order), the TP host arena's measured read (readers a copy, the
 * host role's own cap; piece bytes) and the shape it was measured in (host threads, copy split), enabled 0 for a
 * candidate. False with reason when there is none: missing, not_a_stage_profile (a schema 2 profile, say),
 * no_load_section, load_section_malformed; the caller keeps its defaults. storage_load_section_matches says whether
 * the section names this pack and exactly these copies, by their device keys in order, measured with these host
 * threads and this split (false: load_section_stale_pack, load_section_stale_copies or load_section_stale_shape). */
#define STORAGE_PROFILE_STAGES_SCHEMA 3u
#define STORAGE_LOAD_SECTION_SCHEMA 1u
#define STORAGE_LOAD_SECTION_NAME "load"
#define STORAGE_PACK_IDENTITY_DIGITS \
	16u /* every stage section's header digest, in hexadecimal */
struct storage_load_section {
	char source[REPLICA_PATH_CAPACITY];
	char pack_identity[STORAGE_PROFILE_KEY_BYTES];
	char keys[STORAGE_MAXIMUM_COPIES][STORAGE_PROFILE_KEY_BYTES];
	uint64_t file_byte_count, tp_piece_bytes;
	uint32_t copy_count, enabled, tp_readers_per_copy, tp_host_threads,
		tp_split;
};
bool storage_load_section_read(const char *path,
			       struct storage_load_section *output,
			       char reason[STORAGE_PROFILE_REASON_BYTES]);
bool storage_load_section_matches(const struct storage_load_section *section,
				  const struct pack_header *header,
				  const char *const *copy_paths,
				  uint32_t copy_count, bool external,
				  uint32_t host_threads, uint32_t split,
				  char reason[STORAGE_PROFILE_REASON_BYTES]);
/* An EXL3 map's section was measured in serving's --map_reads auto (the load stage reads as auto does: DROPPED for
 * a lone map, DIRECT_DROPPED with replicas): an explicit map_reads (an enum exl3_read_mode, storage_exl3.h) other
 * than that is another shape, load_section_stale_shape; auto (EXL3_READ_MODES or above) matches. */
bool storage_load_section_reads_match(uint32_t copy_count, uint32_t map_reads,
				      char reason[STORAGE_PROFILE_REASON_BYTES]);
/* Knob over profile over default, a parameter at a time: choice holds the knob's or the default's values on entry;
 * an applicable section (NULL: none) replaces each one no knob gave, and says so. The section's readers were
 * measured on the host arena alone, so they go to tp_host_readers_per_copy (the host role's cap) and the whole load's
 * cap tp_readers_per_copy keeps its knob or default. Without a section nothing changes, so a load without a profile
 * reads exactly as before. */
struct storage_load_choice {
	uint64_t tp_piece_bytes;
	uint32_t tp_readers_per_copy, tp_host_readers_per_copy;
	bool readers_from_profile, piece_from_profile;
};
void storage_load_choose(const struct storage_load_section *section,
			 bool explicit_readers, bool explicit_piece,
			 struct storage_load_choice *choice);
/* The Engram stage's section of a schema 3 profile (anemone autotune PACK --stage=engram;
 * docs/user/STORAGE_PROFILE.md), its own schema: its identity as the load section's (the pack, the copies' device keys
 * in order), the flight shape it was measured in (interface STORAGE_ENGRAM_INTERFACE, steps of slots) and its step I/O
 * selection (threads a copy, spin, pin), enabled 0 for a candidate. storage_engram_section_read: false with reason
 * missing, not_a_stage_profile, no_engram_section or engram_section_malformed. storage_engram_section_matches: whether
 * it names this pack and exactly these copies, by their device keys (storage_device_key_for; an empty key matches
 * none) in order, and this shape (false:
 * engram_section_stale_pack, _stale_copies or _stale_shape). It never holds a RAM share: the tables' place is the RAM
 * fit's. */
#define STORAGE_ENGRAM_SECTION_SCHEMA 1u
#define STORAGE_ENGRAM_INTERFACE "uring_step"
struct storage_engram_section {
	char source[REPLICA_PATH_CAPACITY];
	char pack_identity[STORAGE_PROFILE_KEY_BYTES];
	char keys[STORAGE_MAXIMUM_COPIES][STORAGE_PROFILE_KEY_BYTES];
	uint64_t file_byte_count;
	uint32_t copy_count, enabled, steps, slots, threads_per_copy, spin_us,
		pin;
};
bool storage_engram_section_read(const char *path,
				 struct storage_engram_section *output,
				 char reason[STORAGE_PROFILE_REASON_BYTES]);
bool storage_engram_section_matches(
	const struct storage_engram_section *section,
	const struct pack_header *header,
	const char (*copy_keys)[STORAGE_PROFILE_KEY_BYTES], uint32_t copy_count,
	uint32_t steps, uint32_t slots,
	char reason[STORAGE_PROFILE_REASON_BYTES]);
/* Knob over profile over default, a parameter at a time, as storage_load_choose: choice holds the knob's or the
 * default's values on entry; an applicable section (NULL: none) replaces each one no knob gave, and says so, only while
 * the tables are read from disk (ram_percent below ENGRAM_RAM_PERCENT): with every table in RAM no step I/O starts and
 * the section is not applied. */
struct storage_engram_choice {
	uint32_t threads_per_copy, spin_us, pin;
	bool threads_from_profile, spin_from_profile, pin_from_profile;
};
void storage_engram_choose(const struct storage_engram_section *section,
			   uint32_t ram_percent, bool explicit_threads,
			   bool explicit_spin, bool explicit_pin,
			   struct storage_engram_choice *choice);
/* Serving's application of the profile's engram section where step I/O starts (commands.c's engram_steps_start), its
 * test seam: with the tables on disk (ram_percent below ENGRAM_RAM_PERCENT) the section of the profile at profile
 * (NULL or empty: none) is read, and applied through storage_engram_choose when it is enabled and names this pack
 * (header), exactly the disk store's admitted copies by their device keys in order (copy_keys; an empty key matches
 * none) and serving's flight shape (steps of slots); each knob given (explicit_*) keeps choice's value. reason gets
 * why it is not applied (profile_off, tables_in_ram, the read's or the match's reason, engram_section_candidate),
 * empty when it is. With every table in RAM nothing is read. The RAM share is an input only: it stays the RAM fit's. */
struct storage_engram_serving {
	const char *profile;
	const struct pack_header *header;
	const char (*copy_keys)[STORAGE_PROFILE_KEY_BYTES];
	uint32_t copy_count, steps, slots, ram_percent;
	bool explicit_threads, explicit_spin, explicit_pin;
};
void storage_engram_section_apply(const struct storage_engram_serving *serving,
				  struct storage_engram_choice *choice,
				  char reason[STORAGE_PROFILE_REASON_BYTES]);
/* The schema 3 container, a stage's member by name (STORAGE_LOAD_SECTION_NAME or STORAGE_ENGRAM_SECTION_NAME; any
 * other is STATUS_INVALID_ARGUMENT): file plumbing only, so every stage's parameters, state and decision stay its own.
 * storage_stage_section_write writes section (length bytes of JSON) as that member of the profile at path, atomically:
 * a new schema 3 profile when there is none, else the member replaced or added with every other byte kept. A file that
 * is no schema 3 profile of real_source is refused, its reason in *refusal, so a stage never replaces another tuning's
 * profile. storage_stage_section_qualify turns that member's candidate (enabled 0) into a qualified one (enabled 1),
 * alone. storage_stage_section_writable is read-only, before a stage measures: STATUS_OK when a write could write path
 * (no file, or a schema 3 profile of real_source), else its refusal in *refusal; the write's own check stays
 * authoritative. The storage_load_section_* calls are these for the load stage's member. */
#define STORAGE_ENGRAM_SECTION_NAME "engram"
MUST_CHECK result_status storage_stage_section_write(
	const char *path, const char *real_source, const char *member_name,
	const char *section, size_t length, const char **refusal);
MUST_CHECK result_status storage_stage_section_qualify(const char *path,
						       const char *real_source,
						       const char *member_name,
						       const char **refusal);
MUST_CHECK result_status storage_stage_section_writable(const char *path,
							const char *real_source,
							const char **refusal);
MUST_CHECK result_status storage_load_section_write(const char *path,
						    const char *real_source,
						    const char *section,
						    size_t length,
						    const char **refusal);
MUST_CHECK result_status storage_load_section_qualify(const char *path,
						      const char *real_source,
						      const char **refusal);
MUST_CHECK result_status storage_load_section_writable(const char *path,
						       const char *real_source,
						       const char **refusal);
/* CPU-only, bounded startup validation. No benchmark, copy, service or hardware modification.
 * Every admitted full-pack replica matches every SHA256 chunk in the referenced source manifest.
 * A stale or identity-mismatched replica is excluded independently; no remaining replica or invalid
 * profile-wide configuration gives false and reason.
 * Missing/invalid profiles leave output untouched. Caller chooses its unchanged defaults on false. */
bool storage_profile_read(const char *path, struct storage_profile *output,
			  char reason[STORAGE_PROFILE_REASON_BYTES]);
MUST_CHECK result_status storage_device_key(const char *path,
					    char key[STORAGE_PROFILE_KEY_BYTES],
					    uint64_t *physical_out);
MUST_CHECK result_status storage_device_key_for(
	const char *path, bool external, char key[STORAGE_PROFILE_KEY_BYTES],
	struct local_device_description *description);
enum storage_replica_role { STORAGE_REPLICA_LOAD, STORAGE_REPLICA_LOOKUP };
/* Startup-only observation updates; never called by the HTTP poller or token path. A full table drops a new
 * mount's entry (STATUS_OK): diagnostics never fail startup. */
void storage_profile_roles(bool tuned_load, bool tuned_lookup,
			   bool disk_lookup);
MUST_CHECK result_status
storage_profile_note(const struct local_device_description *device,
		     const char *reason, enum storage_replica_role role);
/* Startup/report diagnostics survive safe configuration fallback. Never mutate after serving starts. */
const struct storage_profile *storage_profile_decisions(void);
/* The copies a tuned load reads, in profile order, with their positive load shares; returns their count. A zero
 * load share keeps a device for lookup only. */
uint32_t storage_profile_load_list(const struct storage_profile *profile,
				   const char *paths[STORAGE_MAXIMUM_COPIES],
				   uint32_t weights[STORAGE_MAXIMUM_COPIES]);
/* The pack header at the start of a pack's bytes, and how many bytes it has, as the loader reads them: an EXL3 map's
 * logical header and size through the map reader (the map file itself starts with the map's own header); any other
 * regular file's first bytes and size. False, outputs untouched, when it does not open or read. */
MUST_CHECK bool storage_pack_header_read(const char *path,
					 struct pack_header *header_out,
					 uint64_t *bytes_out);
/* The profile's disk lookup sources as an --engram_sources list (paths joined by commas) into output: every verified
 * device but an EXL3 map, which the disk store cannot open (it reads packs: the map's flat copies hold its tables);
 * *count_out: how many. False when they do not fit in capacity: output then holds an empty list. */
MUST_CHECK bool
storage_profile_engram_sources(const struct storage_profile *profile,
			       char *output, size_t capacity,
			       uint32_t *count_out);
/* An untuned startup's copy, which the loader reads whatever its media: an included line and note, and a media
 * refusal as one advisory line. */
MUST_CHECK result_status storage_profile_note_loaded(
	FILE *output, const struct local_device_description *device,
	const char *refusal, uint32_t floor_percent);
/* Whether a command-line argument sets the disk lookup (Engram storage) family, which overrides its profile:
 * --engram_io, --engram_ram_percent, --engram_cache_mib, --engram_sources, --turbo-nvme-engrams. */
bool storage_profile_engram_argument(const char *argument);
#define STORAGE_PROFILE_METRICS_BYTES (2u << 20u)
MUST_CHECK result_status storage_profile_metrics(
	FILE *output, const struct storage_profile *profile, bool enabled);
int storage_profile_cli(int count, char **arguments);
#endif
