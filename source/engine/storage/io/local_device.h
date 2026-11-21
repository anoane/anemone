/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef LOCAL_DEVICE_H
#define LOCAL_DEVICE_H
#include <stdbool.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include "common/status.h"
#define LOCAL_DEVICE_MODEL_BYTES 128u
#define LOCAL_DEVICE_TRANSPORT_BYTES 64u
#define LOCAL_DEVICE_FILESYSTEM_BYTES 64u
#define LOCAL_DEVICE_PHYSICAL_MOST 16u
#define LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT 20u
#define LOCAL_DEVICE_PERCENT 100u
/* Physical leaves are shared by partitions, dm/md volumes and bind mounts. */
struct local_device_description {
	uint64_t physical[LOCAL_DEVICE_PHYSICAL_MOST];
	char sysfs[LOCAL_DEVICE_PHYSICAL_MOST][PATH_MAX];
	uint64_t total_bytes, available_bytes;
	char mount[PATH_MAX], model[LOCAL_DEVICE_MODEL_BYTES],
		transport[LOCAL_DEVICE_TRANSPORT_BYTES],
		filesystem[LOCAL_DEVICE_FILESYSTEM_BYTES];
	uint32_t count;
	bool network, virtual_media, external, rotational, readonly;
};
enum local_device_reason {
	REPLICA_ELIGIBLE,
	REPLICA_VERIFIED,
	REPLICA_NETWORK,
	REPLICA_VIRTUAL,
	REPLICA_UNKNOWN_TOPOLOGY,
	REPLICA_EXTERNAL,
	REPLICA_ROTATIONAL,
	REPLICA_READONLY,
	REPLICA_INVALID_SPACE,
	REPLICA_SPACE,
	REPLICA_INVALID_DEVICE,
	REPLICA_DISAPPEARED,
	REPLICA_ID_UNAVAILABLE,
	REPLICA_ID_CHANGED,
	REPLICA_DUPLICATE,
	REPLICA_ZERO_WEIGHT,
	REPLICA_UNVERIFIED,
	REPLICA_RETUNE,
	REPLICA_DIRECTORY,
	REPLICA_FINAL_EXISTS,
	REPLICA_VERIFY_FAILED,
	REPLICA_LIMIT,
	REPLICA_COPY_ERROR,
	REPLICA_COPY_DEVICE_ERROR,
	REPLICA_FINAL_UNVERIFIED,
	REPLICA_DISABLED,
	REPLICA_INVALID_DECISION,
	REPLICA_AUTOMOUNT_NOT_MOUNTED,
	REPLICA_FOREIGN_PARTIAL_KEPT,
	REPLICA_COPY_IN_PROGRESS,
	REPLICA_REASON_COUNT
};
const char *local_device_reason_name(enum local_device_reason reason);
bool local_device_reason_known(const char *reason);
/* Always a product line, including release. No LOG_* macros and no untrusted reason text. */
void local_device_print(FILE *output,
			const struct local_device_description *device,
			const char *reason, uint64_t bytes,
			uint32_t floor_percent, uint32_t load_weight,
			uint32_t lookup_weight);
bool local_device_network_filesystem(const char *name);
bool local_device_external_bus(const char *bus);
/* Resolve a bounded sysfs block topology; filesystem/media admission remains a separate policy check. */
MUST_CHECK result_status local_device_resolve(
	const char *sysfs_path, struct local_device_description *output);
/* The mount entry of mounts (a mount table stream) whose point contains path, the last listed of stacked ones:
 * its raw point and its filesystem type. */
void local_device_mount_of(FILE *mounts, const char *path,
			   struct local_device_description *description);
/* text copied for display (stderr or JSON): control, non-ASCII, quote and backslash bytes become '_'. */
void local_device_display(char *output, size_t capacity, const char *text);
/* Read-only Linux discovery. No probe, helper process or machine changes. */
MUST_CHECK result_status local_device_describe(
	const char *path, struct local_device_description *output);
/* local_device_describe with its mount entries read from the table file mount_table (the product's is
 * /proc/self/mounts); the device topology still comes from sysfs. */
MUST_CHECK result_status
local_device_describe_mounts(const char *mount_table, const char *path,
			     struct local_device_description *output);
/* Pure policy functions also used by fixture tests. NULL means admitted. Network/virtual media never qualify. */
const char *
local_device_media_refusal(const struct local_device_description *device,
			   bool external);
const char *local_device_space_refusal(uint64_t total, uint64_t available,
				       uint64_t bytes, uint32_t floor_percent,
				       uint64_t floor_bytes, bool readonly);
bool local_device_overlaps(const struct local_device_description *left,
			   const struct local_device_description *right);
/* One product line naming both paths and the shared physical identity of a duplicate replica. */
void local_device_print_duplicate(
	FILE *output, const char *excluded_path,
	const struct local_device_description *excluded, const char *kept_path,
	const struct local_device_description *kept);
uint64_t local_device_group(const struct local_device_description *device);
MUST_CHECK result_status local_device_identity(const char *path,
					       uint64_t *physical_out);
MUST_CHECK result_status local_device_copy_room(const char *path,
						uint64_t bytes);
/* Bounded read-only mount scan; NULL json prints only the live product table to stderr. */
MUST_CHECK result_status local_device_scan(FILE *json, bool external);
#define LOCAL_DEVICE_CANDIDATES_MOST 64u
MUST_CHECK result_status
local_device_candidates(bool external, struct local_device_description *devices,
			uint32_t capacity, uint32_t *count);
/* Bounded O_DIRECT media-admission sample, not a sustained/random workload benchmark. */
#define LOCAL_DEVICE_MINIMUM_MIB_PER_SECOND 200u
MUST_CHECK result_status local_device_read_probe(const char *path,
						 uint64_t *mib_per_second_out);
#endif
