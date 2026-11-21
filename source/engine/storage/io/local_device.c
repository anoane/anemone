/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "local_device.h"
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/statvfs.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#include <time.h>
#include <linux/magic.h>
#include <mntent.h>
#include <inttypes.h>

#define DEVICE_PATH_BYTES PATH_MAX
#define DECIMAL_BASE 10
#define DEVICE_NUMBER_PARTS 2
#define HASH_OFFSET 14695981039346656037ull
#define HASH_MULTIPLIER 1099511628211ull
#define MOST_ANCESTORS 32u
#define MOST_TOPOLOGY_NODES 128u
#define UDEV_LINES_MOST 512u
#define MOUNT_ENTRIES_MOST 4096u
#define MOUNT_ENTRY_BYTES \
	(2u *             \
	 DEVICE_PATH_BYTES) /* getmntent_r's strings: a mount's source and point */
#define NETWORK_9P_MAGIC 0x01021997ul
#define NETWORK_CEPH_MAGIC 0x00c36400ul
#define NETWORK_SMB_MAGIC 0x517bul
#define NETWORK_SMB2_MAGIC 0xfe534d42ul
#define NETWORK_CIFS_MAGIC 0xff534d42ul
#define NETWORK_FUSE_MAGIC 0x65735546ul

static const char *const reason_names[REPLICA_REASON_COUNT] = {
	"eligible",
	"verified_identity_and_bytes",
	"network_filesystem_or_transport",
	"virtual_filesystem_or_device",
	"unknown_physical_topology",
	"external_device_requires_override",
	"rotational_device",
	"read_only_filesystem",
	"invalid_space_policy",
	"insufficient_space",
	"invalid_device",
	"device_disappeared_or_unknown",
	"hardware_identity_unavailable",
	"hardware_identity_mismatch_retune_required",
	"duplicate_physical_device",
	"zero_tuned_weight",
	"replica_missing_short_or_unverified",
	"previously_excluded_retune_required",
	"destination_directory_policy",
	"unverified_final_exists",
	"verification_failed",
	"device_limit",
	"copy_io_error",
	"copy_device_io_failure",
	"finalized_replica_unverified",
	"action_disabled",
	"invalid_decision",
	"automount_not_mounted",
	"foreign_partial_kept",
	"copy_in_progress"
};

const char *local_device_reason_name(enum local_device_reason reason)
{
	return (unsigned)reason < REPLICA_REASON_COUNT ?
		       reason_names[reason] :
		       reason_names[REPLICA_INVALID_DECISION];
}

bool local_device_reason_known(const char *reason)
{
	if (reason != NULL)
		for (unsigned index = 0; index < REPLICA_REASON_COUNT; index++)
			if (strcmp(reason, reason_names[index]) == 0)
				return true;
	return false;
}

static void display_text(char *text)
{
	for (; *text != '\0'; text++)
		if ((unsigned char)*text < ' ' || (unsigned char)*text > '~' ||
		    *text == '"' || *text == '\\')
			*text = '_';
}

void local_device_print(FILE *output,
			const struct local_device_description *device,
			const char *reason, uint64_t bytes,
			uint32_t floor_percent, uint32_t load_weight,
			uint32_t lookup_weight)
{
	if (output == NULL || device == NULL)
		return;
	char mount[PATH_MAX], model[LOCAL_DEVICE_MODEL_BYTES],
		transport[LOCAL_DEVICE_TRANSPORT_BYTES],
		filesystem[LOCAL_DEVICE_FILESYSTEM_BYTES];
	(void)snprintf(mount, sizeof(mount), "%.*s", (int)sizeof(mount) - 1,
		       device->mount);
	(void)snprintf(model, sizeof(model), "%.*s", (int)sizeof(model) - 1,
		       device->model);
	(void)snprintf(transport, sizeof(transport), "%.*s",
		       (int)sizeof(transport) - 1, device->transport);
	(void)snprintf(filesystem, sizeof(filesystem), "%.*s",
		       (int)sizeof(filesystem) - 1, device->filesystem);
	display_text(mount);
	display_text(model);
	display_text(transport);
	display_text(filesystem);
	const char *const fixed =
		reason == NULL ? reason_names[REPLICA_VERIFIED] :
		local_device_reason_known(reason) ?
				 reason :
				 reason_names[REPLICA_INVALID_DECISION];
	const bool included =
		strcmp(fixed, reason_names[REPLICA_VERIFIED]) == 0 ||
		strcmp(fixed, reason_names[REPLICA_ELIGIBLE]) == 0;
	(void)fprintf(output,
		      "storage: replica %s (%s, %s, %016" PRIx64 ") %s: %s",
		      mount[0] != '\0' ? mount : "unavailable",
		      model[0] != '\0' ? model : "unknown",
		      transport[0] != '\0' ? transport : "unknown",
		      local_device_group(device),
		      included ? "included" : "excluded", fixed);
	if (included)
		(void)fprintf(output, " load_weight=%u lookup_weight=%u",
			      load_weight, lookup_weight);
	else if (strcmp(fixed, reason_names[REPLICA_SPACE]) == 0)
		(void)fprintf(output,
			      " needs_bytes=%" PRIu64
			      " floor_percent=%u available_bytes=%" PRIu64,
			      bytes, floor_percent, device->available_bytes);
	else if (strcmp(fixed, reason_names[REPLICA_EXTERNAL]) == 0)
		(void)fputs(" (use --replica_external=1)", output);
	else if (strcmp(fixed, reason_names[REPLICA_NETWORK]) == 0)
		(void)fprintf(output, " filesystem=%s",
			      filesystem[0] != '\0' ? filesystem : "unknown");
	(void)fputc('\n', output);
}

void local_device_mount_of(FILE *mounts, const char *path,
			   struct local_device_description *description)
{
	struct mntent entry, *found;
	char buffer[MOUNT_ENTRY_BYTES];
	size_t longest = 0;
	uint32_t entries = 0;
	while (entries++ < MOUNT_ENTRIES_MOST &&
	       (found = getmntent_r(mounts, &entry, buffer, sizeof(buffer))) !=
		       NULL) {
		const size_t length = strlen(found->mnt_dir);
		/* >=: of stacked entries on one point the last listed is the visible one (an nfs4 over its autofs) */
		if (length >= longest &&
		    strncmp(path, found->mnt_dir, length) == 0 &&
		    (length == 1u || path[length] == '/' ||
		     path[length] == '\0')) {
			(void)snprintf(description->mount,
				       sizeof(description->mount), "%s",
				       found->mnt_dir);
			(void)snprintf(description->filesystem,
				       sizeof(description->filesystem), "%s",
				       found->mnt_type);
			longest = length;
		}
	}
	display_text(
		description
			->filesystem); /* the point stays the raw path: printers sanitise their copies */
}

void local_device_display(char *output, size_t capacity, const char *text)
{
	(void)snprintf(output, capacity, "%.*s", (int)capacity - 1, text);
	display_text(output);
}

static void describe_mount(const char *mount_table, const char *path,
			   struct local_device_description *description)
{
	FILE *const mounts = setmntent(mount_table, "r");
	if (mounts == NULL)
		return;
	local_device_mount_of(mounts, path, description);
	(void)endmntent(mounts);
}

static bool text_read(const char *directory, const char *name, char *text,
		      size_t capacity)
{
	char path[DEVICE_PATH_BYTES];
	const int length =
		snprintf(path, sizeof(path), "%s/%s", directory, name);
	if (length < 0 || (size_t)length >= sizeof(path))
		return false;
	const int descriptor = open(path, O_RDONLY | O_CLOEXEC);
	if (descriptor < 0)
		return false;
	const ssize_t count = read(descriptor, text, capacity - 1);
	(void)close(descriptor);
	if (count <= 0 || (size_t)count >= capacity - 1)
		return false;
	text[count] = '\0';
	return true;
}

static int device_flag(const char *directory, const char *name)
{
	char text[DEVICE_PATH_BYTES];
	if (!text_read(directory, name, text, sizeof(text)))
		return -1;
	char *end = NULL;
	errno = 0;
	const long value = strtol(text, &end, DECIMAL_BASE);
	return errno == 0 && end != text && (*end == '\n' || *end == '\0') &&
			       value >= 0 && value <= INT_MAX ?
		       (int)value :
		       -1;
}

static uint64_t path_hash(const char *path)
{
	uint64_t value = HASH_OFFSET;
	for (const unsigned char *byte = (const unsigned char *)path;
	     *byte != '\0'; byte++)
		value = (value ^ *byte) * HASH_MULTIPLIER;
	return value;
}

/* An identity text must name the device: a blank serial would merge distinct drives into one identity. */
static bool identity_read(const char *directory, const char *name,
			  char text[DEVICE_PATH_BYTES])
{
	return text_read(directory, name, text, DEVICE_PATH_BYTES) &&
	       text[strspn(text, " \t\r\n")] != '\0';
}

/* The PCI function above nvme/nvmeN: stable when the kernel renumbers the controller, shared by its namespaces. */
static void nvme_function(char controller[DEVICE_PATH_BYTES])
{
	char *slash = strrchr(controller, '/');
	if (slash == NULL || slash == controller)
		return;
	*slash = '\0';
	slash = strrchr(controller, '/');
	if (slash != NULL && slash != controller &&
	    strcmp(slash + 1, "nvme") == 0)
		*slash = '\0';
}

bool local_device_external_bus(const char *bus)
{
	return bus != NULL &&
	       (strcmp(bus, "usb") == 0 || strcmp(bus, "mmc") == 0 ||
		strcmp(bus, "thunderbolt") == 0 ||
		strcmp(bus, "ieee1394") == 0);
}

static bool udev_external(const char *device)
{
	char number[DEVICE_PATH_BYTES], path[DEVICE_PATH_BYTES];
	unsigned major_number = 0, minor_number = 0;
	if (!text_read(device, "dev", number, sizeof(number)) ||
	    sscanf(number, "%u:%u", &major_number, &minor_number) !=
		    DEVICE_NUMBER_PARTS)
		return false;
	const int length = snprintf(path, sizeof(path), "/run/udev/data/b%u:%u",
				    major_number, minor_number);
	if (length < 0 || (size_t)length >= sizeof(path))
		return false;
	FILE *const file = fopen(path, "re");
	if (file == NULL)
		return false; /* sysfs removable and ancestry remain authoritative when udev is absent */
	bool external = false;
	for (uint32_t line = 0; line < UDEV_LINES_MOST &&
				fgets(number, sizeof(number), file) != NULL;
	     line++)
		if (strncmp(number, "E:ID_BUS=", strlen("E:ID_BUS=")) == 0) {
			number[strcspn(number, "\r\n")] = '\0';
			external = external ||
				   local_device_external_bus(
					   number + strlen("E:ID_BUS="));
		}
	(void)fclose(file);
	return external;
}

static bool udev_identity(const char *device, char output[DEVICE_PATH_BYTES])
{
	char number[DEVICE_PATH_BYTES], path[DEVICE_PATH_BYTES];
	unsigned major_number = 0, minor_number = 0;
	if (!text_read(device, "dev", number, sizeof(number)) ||
	    sscanf(number, "%u:%u", &major_number, &minor_number) !=
		    DEVICE_NUMBER_PARTS)
		return false;
	const int length = snprintf(path, sizeof(path), "/run/udev/data/b%u:%u",
				    major_number, minor_number);
	if (length < 0 || (size_t)length >= sizeof(path))
		return false;
	FILE *const file = fopen(path, "re");
	if (file == NULL)
		return false;
	bool found = false;
	for (uint32_t line = 0; line < UDEV_LINES_MOST &&
				fgets(number, sizeof(number), file) != NULL;
	     line++) {
		if (strncmp(number, "E:ID_SERIAL=", strlen("E:ID_SERIAL=")) !=
		    0)
			continue;
		(void)snprintf(output, DEVICE_PATH_BYTES, "%s", number);
		found = true;
		break;
	}
	(void)fclose(file);
	return found;
}

struct topology_node {
	char path[DEVICE_PATH_BYTES];
	uint32_t depth;
};

/* One leaf block device (no slaves): its media flags and its physical identity, kept sorted and once. */
static bool leaf_physical(const char *device,
			  struct local_device_description *result)
{
	const char *const name = strrchr(device, '/');
	if (name == NULL || strstr(device, "/virtual/") != NULL ||
	    strstr(device, "/virtio") != NULL) {
		result->virtual_media = true;
		return true;
	}
	const bool nvme = strncmp(name + 1, "nvme", strlen("nvme")) == 0;
	const bool sd = strncmp(name + 1, "sd", strlen("sd")) == 0;
	const bool mmc = strncmp(name + 1, "mmcblk", strlen("mmcblk")) == 0;
	if (!nvme && !sd && !mmc)
		return false;
	const int rotational = device_flag(device, "queue/rotational");
	const int removable = device_flag(device, "removable");
	if (rotational < 0 || removable < 0)
		return false;
	result->external = result->external || removable != 0 || mmc ||
			   strstr(device, "/usb") != NULL ||
			   udev_external(device);
	result->rotational = result->rotational || rotational != 0;
	result->network = result->network ||
			  strstr(device, "/session") != NULL ||
			  strstr(device, "/iscsi") != NULL;
	char group[DEVICE_PATH_BYTES];
	memcpy(group, device, strlen(device) + 1);
	if (nvme) {
		char *const slash = strrchr(group, '/');
		if (slash == NULL)
			return false;
		*slash = '\0';
		char transport[DEVICE_PATH_BYTES];
		if (text_read(group, "transport", transport,
			      sizeof(transport)) &&
		    strcmp(transport, "pcie\n") != 0)
			result->network = true;
	}
	char ancestor[DEVICE_PATH_BYTES], placement[DEVICE_PATH_BYTES];
	memcpy(ancestor, device, strlen(device) + 1);
	for (uint32_t level = 0; level < MOST_ANCESTORS; level++) {
		/* PCI devices below an external-facing port (Thunderbolt/USB4 tunnels) carry the text attribute
		 * removable = "removable"; a block device's own removable is the integer read above. */
		result->external = result->external ||
				   strstr(ancestor, "/thunderbolt") != NULL ||
				   (text_read(ancestor, "removable", placement,
					      sizeof(placement)) &&
				    strcmp(placement, "removable\n") == 0);
		char *const slash = strrchr(ancestor, '/');
		if (slash == NULL || slash == ancestor)
			break;
		*slash = '\0';
	}
	char identity_text[DEVICE_PATH_BYTES], model[DEVICE_PATH_BYTES];
	uint64_t identity = path_hash(group);
	/* NVMe identity is the controller's, never a namespace's wwid or the controller's kernel name. */
	if (nvme && identity_read(group, "serial", identity_text)) {
		identity = path_hash(identity_text);
		if (text_read(group, "model", model, sizeof(model)))
			identity ^= path_hash(model);
	} else if (nvme) {
		nvme_function(group);
		identity = path_hash(group);
	} else if (identity_read(device, "wwid", identity_text) ||
		   identity_read(device, "device/wwid", identity_text) ||
		   identity_read(device, "device/cid", identity_text) ||
		   udev_identity(device, identity_text)) {
		identity = path_hash(identity_text);
	}
	for (uint32_t index = 0; index < result->count; index++)
		if (result->physical[index] == identity)
			return true;
	if (result->count == LOCAL_DEVICE_PHYSICAL_MOST)
		return false;
	uint32_t position = result->count;
	while (position != 0 && result->physical[position - 1u] > identity) {
		result->physical[position] = result->physical[position - 1u];
		memcpy(result->sysfs[position], result->sysfs[position - 1u],
		       sizeof(result->sysfs[position]));
		position--;
	}
	result->physical[position] = identity;
	memcpy(result->sysfs[position], device, strlen(device) + 1);
	result->count++;
	return true;
}

/* The physical leaves under path, iteratively: a bounded work list of at most MOST_TOPOLOGY_NODES sysfs nodes,
 * each at most MOST_ANCESTORS slave levels deep (no recursion, so the stack stays bounded). */
static bool collect_physical(const char *path,
			     struct local_device_description *result)
{
	struct topology_walk {
		struct topology_node pending[MOST_TOPOLOGY_NODES];
		char device[DEVICE_PATH_BYTES], children[DEVICE_PATH_BYTES];
	} *const walk = malloc(sizeof(struct topology_walk));
	if (walk == NULL)
		return false;
	uint32_t count = 0, visited = 0;
	bool valid = snprintf(walk->pending[0].path,
			      sizeof(walk->pending[0].path), "%s",
			      path) < (int)sizeof(walk->pending[0].path);
	walk->pending[0].depth = 0;
	count = valid ? 1u : 0u;
	while (valid && count != 0) {
		const uint32_t depth = walk->pending[--count].depth;
		valid = visited++ < MOST_TOPOLOGY_NODES &&
			depth < MOST_ANCESTORS &&
			realpath(walk->pending[count].path, walk->device) !=
				NULL;
		if (valid && device_flag(walk->device, "partition") >= 0) {
			char *const slash = strrchr(walk->device, '/');
			valid = slash != NULL;
			if (valid)
				*slash = '\0';
		}
		const int length = valid ? snprintf(walk->children,
						    sizeof(walk->children),
						    "%s/slaves", walk->device) :
					   -1;
		valid = valid && length >= 0 &&
			(size_t)length < sizeof(walk->children);
		DIR *const slaves = valid ? opendir(walk->children) : NULL;
		uint32_t found = 0;
		struct dirent *entry;
		while (slaves != NULL && valid &&
		       (entry = readdir(slaves)) != NULL) {
			if (entry->d_name[0] == '.')
				continue;
			/* every pending node is visited later: the walk stays within the node bound */
			valid = count < MOST_TOPOLOGY_NODES - visited;
			struct topology_node *const child =
				&walk->pending[count];
			const int added = valid ? snprintf(child->path,
							   sizeof(child->path),
							   "%s/%s",
							   walk->children,
							   entry->d_name) :
						  -1;
			valid = valid && added > 0 &&
				(size_t)added < sizeof(child->path);
			if (valid) {
				child->depth = depth + 1u;
				count++;
			}
			found++;
		}
		if (slaves != NULL)
			(void)closedir(slaves);
		if (valid && found == 0)
			valid = leaf_physical(walk->device, result);
	}
	free(walk);
	return valid;
}

result_status local_device_resolve(const char *sysfs_path,
				   struct local_device_description *output)
{
	if (sysfs_path == NULL || output == NULL)
		return STATUS_INVALID_ARGUMENT;
	struct local_device_description result = { 0 };
	if (!collect_physical(sysfs_path, &result))
		return STATUS_UNSUPPORTED;
	*output = result;
	return STATUS_OK;
}

bool local_device_network_filesystem(const char *name)
{
	return name != NULL &&
	       (strcmp(name, "nfs") == 0 || strcmp(name, "nfs4") == 0 ||
		strcmp(name, "cifs") == 0 || strcmp(name, "smb3") == 0 ||
		strcmp(name, "smbfs") == 0 || strcmp(name, "9p") == 0 ||
		strcmp(name, "ceph") == 0 || strcmp(name, "glusterfs") == 0 ||
		strcmp(name, "sshfs") == 0 ||
		strncmp(name, "fuse", strlen("fuse")) == 0);
}

result_status local_device_describe(const char *path,
				    struct local_device_description *output)
{
	return local_device_describe_mounts("/proc/self/mounts", path, output);
}

result_status
local_device_describe_mounts(const char *mount_table, const char *path,
			     struct local_device_description *output)
{
	if (mount_table == NULL || path == NULL || output == NULL)
		return STATUS_INVALID_ARGUMENT;
	struct local_device_description mounted = { 0 };
	describe_mount(mount_table, path, &mounted);
	const char *const type_name = mounted.filesystem;
	/* The visible entry is an automount that has not fired: a stat below its point would mount it (an NFS
	   lookup, or a wait for the mount timeout with the server down). Nothing is called on the path. */
	if (strcmp(type_name, "autofs") == 0) {
		*output =
			mounted; /* its point and filesystem name the refusal */
		return STATUS_UNSUPPORTED;
	}
	if (local_device_network_filesystem(type_name)) {
		mounted.network = true;
		(void)snprintf(mounted.transport, sizeof(mounted.transport),
			       "%s", "network");
		*output = mounted;
		return STATUS_OK;
	}
	struct stat file;
	struct statfs filesystem;
	struct statvfs space;
	if (stat(path, &file) != 0 || statfs(path, &filesystem) != 0 ||
	    statvfs(path, &space) != 0)
		return STATUS_INPUT_OUTPUT_ERROR;
	if ((!S_ISREG(file.st_mode) && !S_ISDIR(file.st_mode)) ||
	    space.f_frsize == 0 ||
	    space.f_blocks > UINT64_MAX / space.f_frsize ||
	    space.f_bavail > UINT64_MAX / space.f_frsize)
		return STATUS_UNSUPPORTED;
	struct local_device_description result = { 0 };
	result.total_bytes = (uint64_t)space.f_blocks * space.f_frsize;
	result.available_bytes = (uint64_t)space.f_bavail * space.f_frsize;
	result.readonly = (space.f_flag & ST_RDONLY) != 0;
	const unsigned long type = (unsigned long)filesystem.f_type;
	result.network =
		type == NFS_SUPER_MAGIC || type == NETWORK_9P_MAGIC ||
		type == NETWORK_CEPH_MAGIC || type == NETWORK_SMB_MAGIC ||
		type == NETWORK_SMB2_MAGIC || type == NETWORK_CIFS_MAGIC ||
		type == NETWORK_FUSE_MAGIC;
	result.virtual_media = type == TMPFS_MAGIC || type == RAMFS_MAGIC ||
			       type == OVERLAYFS_SUPER_MAGIC ||
			       type == PROC_SUPER_MAGIC || type == SYSFS_MAGIC;
	if (!result.network && !result.virtual_media) {
		char link[DEVICE_PATH_BYTES];
		const int length =
			snprintf(link, sizeof(link), "/sys/dev/block/%u:%u",
				 major(file.st_dev), minor(file.st_dev));
		if (length < 0 || (size_t)length >= sizeof(link) ||
		    !collect_physical(link, &result))
			return STATUS_UNSUPPORTED;
	}
	char canonical[DEVICE_PATH_BYTES];
	describe_mount(mount_table,
		       realpath(path, canonical) != NULL ? canonical : path,
		       &result);
	(void)snprintf(result.transport, sizeof(result.transport), "%s",
		       result.network	    ? "network" :
		       result.virtual_media ? "virtual" :
		       result.external	    ? "external" :
					      "internal");
	if (result.count != 0) {
		char model[DEVICE_PATH_BYTES], leaf[DEVICE_PATH_BYTES];
		memcpy(leaf, result.sysfs[0], sizeof(leaf));
		bool read =
			text_read(leaf, "device/model", model, sizeof(model));
		char *const slash = strrchr(leaf, '/');
		if (!read && slash != NULL) {
			*slash = '\0';
			read = text_read(leaf, "model", model, sizeof(model));
		}
		if (read) {
			model[strcspn(model, "\r\n")] = '\0';
			(void)snprintf(result.model, sizeof(result.model),
				       "%.*s", (int)sizeof(result.model) - 1,
				       model);
			display_text(result.model);
		}
		(void)snprintf(
			result.transport, sizeof(result.transport), "%s",
			strstr(result.sysfs[0], "/usb") != NULL	 ? "usb" :
			strstr(result.sysfs[0], "/nvme") != NULL ? "nvme" :
			strstr(result.sysfs[0], "/mmc") != NULL	 ? "mmc" :
			result.external				 ? "removable" :
								   "scsi_ata");
	}
	*output = result;
	return STATUS_OK;
}

const char *
local_device_media_refusal(const struct local_device_description *device,
			   bool external)
{
	return device == NULL	     ? "invalid_device" :
	       device->network	     ? "network_filesystem_or_transport" :
	       device->virtual_media ? "virtual_filesystem_or_device" :
	       device->count == 0 ||
			       device->count > LOCAL_DEVICE_PHYSICAL_MOST ?
				       "unknown_physical_topology" :
	       device->external && !external ?
				       "external_device_requires_override" :
	       device->rotational && !(device->external && external) ?
				       "rotational_device" :
				       NULL;
}

const char *local_device_space_refusal(uint64_t total, uint64_t available,
				       uint64_t bytes, uint32_t floor_percent,
				       uint64_t floor_bytes, bool readonly)
{
	if (readonly)
		return "read_only_filesystem";
	if (total == 0 || available > total ||
	    floor_percent >= LOCAL_DEVICE_PERCENT)
		return "invalid_space_policy";
	const uint64_t fraction =
		total / LOCAL_DEVICE_PERCENT * floor_percent +
		(total % LOCAL_DEVICE_PERCENT * floor_percent +
		 LOCAL_DEVICE_PERCENT - 1u) /
			LOCAL_DEVICE_PERCENT;
	const uint64_t floor = fraction > floor_bytes ? fraction : floor_bytes;
	return available < floor || bytes > available - floor ?
		       "insufficient_space" :
		       NULL;
}

static bool shared_physical(const struct local_device_description *left,
			    const struct local_device_description *right,
			    uint64_t *shared)
{
	if (left == NULL || right == NULL)
		return false;
	for (uint32_t first = 0;
	     first < left->count && first < LOCAL_DEVICE_PHYSICAL_MOST; first++)
		for (uint32_t second = 0; second < right->count &&
					  second < LOCAL_DEVICE_PHYSICAL_MOST;
		     second++)
			if (left->physical[first] == right->physical[second]) {
				*shared = left->physical[first];
				return true;
			}
	return false;
}

bool local_device_overlaps(const struct local_device_description *left,
			   const struct local_device_description *right)
{
	uint64_t shared = 0;
	return shared_physical(left, right, &shared);
}

void local_device_print_duplicate(
	FILE *output, const char *excluded_path,
	const struct local_device_description *excluded, const char *kept_path,
	const struct local_device_description *kept)
{
	uint64_t shared = 0;
	if (output == NULL || excluded_path == NULL || kept_path == NULL ||
	    !shared_physical(kept, excluded, &shared))
		return;
	char excluded_text[PATH_MAX], kept_text[PATH_MAX];
	(void)snprintf(excluded_text, sizeof(excluded_text), "%.*s",
		       (int)sizeof(excluded_text) - 1, excluded_path);
	(void)snprintf(kept_text, sizeof(kept_text), "%.*s",
		       (int)sizeof(kept_text) - 1, kept_path);
	display_text(excluded_text);
	display_text(kept_text);
	(void)fprintf(output,
		      "storage: replica %s excluded: duplicate_physical_device"
		      " (shares physical device %016" PRIx64
		      " with kept replica %s)\n",
		      excluded_text, shared, kept_text);
}

uint64_t local_device_group(const struct local_device_description *device)
{
	if (device == NULL || device->count == 0 ||
	    device->count > LOCAL_DEVICE_PHYSICAL_MOST)
		return 0;
	if (device->count == 1u)
		return device->physical[0];
	uint64_t result = HASH_OFFSET;
	for (uint32_t index = 0; index < device->count; index++)
		result = (result ^ device->physical[index]) * HASH_MULTIPLIER;
	return result;
}

result_status local_device_identity(const char *path, uint64_t *physical_out)
{
	if (physical_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	struct local_device_description device;
	const result_status status = local_device_describe(path, &device);
	if (status != STATUS_OK)
		return status;
	if (local_device_media_refusal(&device, false) != NULL)
		return STATUS_UNSUPPORTED;
	*physical_out = local_device_group(&device);
	return STATUS_OK;
}

result_status local_device_copy_room(const char *path, uint64_t bytes)
{
	struct local_device_description device;
	const result_status status = local_device_describe(path, &device);
	if (status != STATUS_OK)
		return status;
	if (local_device_media_refusal(&device, false) != NULL)
		return STATUS_UNSUPPORTED;
	return local_device_space_refusal(device.total_bytes,
					  device.available_bytes, bytes,
					  LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT, 0,
					  device.readonly) == NULL ?
		       STATUS_OK :
		       STATUS_OUT_OF_MEMORY;
}

#define SCAN_MOUNTS_MOST 64u
#define SCAN_ENTRIES_MOST 4096u

MUST_CHECK static result_status
device_scan(FILE *json, bool external, struct local_device_description *devices,
	    uint32_t capacity, uint32_t *count)
{
	uint64_t seen[SCAN_MOUNTS_MOST * LOCAL_DEVICE_PHYSICAL_MOST];
	uint32_t seen_count = 0, candidates = 0;
	dev_t filesystems[SCAN_MOUNTS_MOST];
	uint32_t filesystem_count = 0;
	if (json != NULL)
		(void)fputs("{\"candidates\":[", json);
	/* Physical block mounts precede the diagnostic non-block exclusions, so virtual submounts cannot hide disks. */
	for (uint32_t pass = 0; pass <= 1u; pass++) {
		FILE *const mounts = setmntent("/proc/self/mounts", "r");
		if (mounts == NULL)
			return STATUS_INPUT_OUTPUT_ERROR;
		struct mntent entry, *found;
		char buffer[MOUNT_ENTRY_BYTES];
		uint32_t examined = 0;
		while (candidates < SCAN_MOUNTS_MOST &&
		       examined++ < SCAN_ENTRIES_MOST &&
		       (found = getmntent_r(mounts, &entry, buffer,
					    sizeof(buffer))) != NULL) {
			const bool block = strncmp(found->mnt_fsname, "/dev/",
						   strlen("/dev/")) == 0;
			if ((pass == 0) != block)
				continue;
			struct stat file;
			const bool network = local_device_network_filesystem(
				found->mnt_type);
			if (!network && stat(found->mnt_dir, &file) != 0)
				continue;
			if (network)
				memset(&file, 0, sizeof(file));
			bool seen_filesystem = false;
			for (uint32_t index = 0; index < filesystem_count;
			     index++)
				seen_filesystem = seen_filesystem ||
						  filesystems[index] ==
							  file.st_dev;
			if (!network && !block && seen_filesystem)
				continue;
			if (!network && !seen_filesystem &&
			    filesystem_count < SCAN_MOUNTS_MOST)
				filesystems[filesystem_count++] = file.st_dev;
			struct local_device_description device = { 0 };
			const result_status described =
				local_device_describe(found->mnt_dir, &device);
			const char *refusal =
				described != STATUS_OK ?
					"device_disappeared_or_unknown" :
					local_device_media_refusal(&device,
								   external);
			if (described != STATUS_OK) {
				(void)snprintf(device.mount,
					       sizeof(device.mount), "%s",
					       found->mnt_dir);
			}
			for (uint32_t leaf = 0;
			     refusal == NULL && leaf < device.count; leaf++)
				for (uint32_t previous = 0;
				     previous < seen_count; previous++)
					if (seen[previous] ==
					    device.physical[leaf])
						refusal =
							"duplicate_physical_device";
			if (refusal == NULL)
				for (uint32_t leaf = 0; leaf < device.count;
				     leaf++)
					seen[seen_count++] =
						device.physical[leaf];
			local_device_print(
				stderr, &device,
				refusal != NULL ? refusal : "eligible", 0,
				LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT, 0, 0);
			if (json != NULL) {
				char shown[PATH_MAX];
				local_device_display(shown, sizeof(shown),
						     device.mount);
				(void)fprintf(
					json,
					"%s{\"path\":\"%s\",\"mount\":\"%s\",\"model\":\"%s\",\"transport\":\"%s\","
					"\"filesystem\":\"%s\",\"included\":%u,\"reason\":\"%s\",\"physical\":"
					"\"%016" PRIx64 "\"}",
					candidates == 0 ? "" : ",", shown,
					shown, device.model, device.transport,
					device.filesystem,
					refusal == NULL ? 1u : 0u,
					refusal != NULL ? refusal : "eligible",
					local_device_group(&device));
			}
			if (devices != NULL && candidates < capacity)
				devices[candidates] = device;
			candidates++;
		}
		(void)endmntent(mounts);
	}
	if (count != NULL)
		*count = candidates < capacity ? candidates : capacity;
	if (json != NULL)
		(void)fputs("]}\n", json);
	return json != NULL && ferror(json) ? STATUS_INPUT_OUTPUT_ERROR :
					      STATUS_OK;
}

#define PROBE_ALIGNMENT_BYTES 4096u
#define PROBE_BYTES (1u << 20u)
#define PROBE_READS 4u
#define PROBE_NANOSECONDS_PER_SECOND 1000000000ull

result_status local_device_read_probe(const char *path,
				      uint64_t *mib_per_second_out)
{
	uint64_t identity = 0;
	if (mib_per_second_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	result_status status = local_device_identity(path, &identity);
	if (status != STATUS_OK)
		return status;
	const int descriptor =
		open(path, O_RDONLY | O_CLOEXEC | O_DIRECT | O_NONBLOCK);
	if (descriptor < 0)
		return STATUS_INPUT_OUTPUT_ERROR;
	struct statx file = { 0 };
	void *buffer = NULL;
	if (statx(descriptor, "", AT_EMPTY_PATH,
		  STATX_DIOALIGN | STATX_SIZE | STATX_TYPE, &file) != 0 ||
	    (file.stx_mask & (STATX_DIOALIGN | STATX_SIZE | STATX_TYPE)) !=
		    (STATX_DIOALIGN | STATX_SIZE | STATX_TYPE) ||
	    !S_ISREG(file.stx_mode) ||
	    file.stx_size < (uint64_t)PROBE_BYTES * PROBE_READS ||
	    file.stx_size > INT64_MAX || file.stx_dio_mem_align == 0 ||
	    file.stx_dio_offset_align == 0 ||
	    PROBE_ALIGNMENT_BYTES % file.stx_dio_mem_align != 0 ||
	    PROBE_ALIGNMENT_BYTES % file.stx_dio_offset_align != 0)
		status = STATUS_UNSUPPORTED;
	if (status == STATUS_OK &&
	    posix_memalign(&buffer, PROBE_ALIGNMENT_BYTES, PROBE_BYTES) != 0)
		status = STATUS_OUT_OF_MEMORY;
	struct timespec begin = { 0 }, end = { 0 };
	if (status == STATUS_OK && clock_gettime(CLOCK_MONOTONIC, &begin) != 0)
		status = STATUS_TIMED_OUT;
	for (uint32_t index = 0; status == STATUS_OK && index < PROBE_READS;
	     index++) {
		const uint64_t offset =
			(file.stx_size - PROBE_BYTES) / PROBE_READS * index /
			PROBE_ALIGNMENT_BYTES * PROBE_ALIGNMENT_BYTES;
		if (pread(descriptor, buffer, PROBE_BYTES, (off_t)offset) !=
		    PROBE_BYTES)
			status = STATUS_INPUT_OUTPUT_ERROR;
	}
	if (status == STATUS_OK && clock_gettime(CLOCK_MONOTONIC, &end) != 0)
		status = STATUS_TIMED_OUT;
	free(buffer);
	if (close(descriptor) != 0 && status == STATUS_OK)
		status = STATUS_INPUT_OUTPUT_ERROR;
	if (status != STATUS_OK)
		return status;
	const int64_t seconds = end.tv_sec - begin.tv_sec;
	const int64_t nanoseconds = end.tv_nsec - begin.tv_nsec;
	if (seconds < 0 ||
	    seconds > INT64_MAX / (int64_t)PROBE_NANOSECONDS_PER_SECOND)
		return STATUS_TIMED_OUT;
	const int64_t elapsed =
		seconds * (int64_t)PROBE_NANOSECONDS_PER_SECOND + nanoseconds;
	if (elapsed <= 0)
		return STATUS_TIMED_OUT;
	*mib_per_second_out =
		PROBE_READS * PROBE_NANOSECONDS_PER_SECOND / (uint64_t)elapsed;
	return STATUS_OK;
}

result_status local_device_scan(FILE *json, bool external)
{
	return device_scan(json, external, NULL, 0, NULL);
}
result_status local_device_candidates(bool external,
				      struct local_device_description *devices,
				      uint32_t capacity, uint32_t *count)
{
	if (devices == NULL || count == NULL || capacity == 0 ||
	    capacity > LOCAL_DEVICE_CANDIDATES_MOST)
		return STATUS_INVALID_ARGUMENT;
	uint32_t found = 0;
	const result_status status =
		device_scan(NULL, external, devices, capacity, &found);
	if (status == STATUS_OK)
		*count = found;
	return status;
}
