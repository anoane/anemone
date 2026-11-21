/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ftw.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
#include "engine/storage/io/local_device.h"
static unsigned failures;
static void expect(int passed, const char *message)
{
	if (!passed) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", message);
	}
}

#define TEST_TOTAL_BYTES 1000u
#define TEST_AVAILABLE_BYTES 700u
#define TEST_COPY_BYTES 500u
#define TEST_CUSTOM_FLOOR 10u
#define TEST_SECOND_DEVICE 2u
#define TEST_THIRD_DEVICE 3u
#define TEST_LOAD_WEIGHT 3u
#define TEST_LOOKUP_WEIGHT 7u
#define TEST_SPACE_BYTES 517u
#define TEST_SPACE_AVAILABLE 310u

#define FIXTURE_MODE 0700
#define FIXTURE_DEPTH 32 /* nftw's open directories */
#define FIXTURE_LINE_BYTES 128u /* a mount table line besides its point */
#define FIXTURE_TABLE_BYTES \
	(2u * (PATH_MAX + FIXTURE_LINE_BYTES)) /* two mount table lines */

static bool fixture_directory(const char *root, const char *relative)
{
	char path[PATH_MAX];
	const int length =
		snprintf(path, sizeof(path), "%s/%s", root, relative);
	if (length < 0 || (size_t)length >= sizeof(path))
		return false;
	for (char *cursor = path + strlen(root) + 1u; *cursor != '\0';
	     cursor++) {
		if (*cursor != '/')
			continue;
		*cursor = '\0';
		const bool made = mkdir(path, FIXTURE_MODE) == 0 ||
				  errno == EEXIST;
		*cursor = '/';
		if (!made)
			return false;
	}
	return mkdir(path, FIXTURE_MODE) == 0 || errno == EEXIST;
}

static bool fixture_file(const char *root, const char *relative,
			 const char *text)
{
	char path[PATH_MAX];
	const int length =
		snprintf(path, sizeof(path), "%s/%s", root, relative);
	if (length < 0 || (size_t)length >= sizeof(path))
		return false;
	FILE *const file = fopen(path, "we");
	if (file == NULL)
		return false;
	const bool written = fputs(text, file) >= 0;
	return fclose(file) == 0 && written;
}

/* directory/name in a PATH_MAX buffer; false if it does not fit. */
static bool fixture_join(char path[PATH_MAX], const char *directory,
			 const char *name)
{
	const int length = snprintf(path, PATH_MAX, "%s/%s", directory, name);
	return length > 0 && (size_t)length < PATH_MAX;
}

/* One entry of a fixture tree, removed after its children (nftw walks the tree; nothing here recurses). */
static int fixture_entry_remove(const char *path, const struct stat *file,
				int kind, struct FTW *walk)
{
	(void)file;
	(void)kind;
	(void)walk;
	return remove(path);
}

static bool fixture_remove(const char *path)
{
	return nftw(path, fixture_entry_remove, FIXTURE_DEPTH,
		    FTW_DEPTH | FTW_PHYS) == 0;
}

#define CHAIN_LEVELS 40u /* deeper than the walk's 32-level bound */
/* virtual/chain/levelN/slaves/next -> levelN+1, the last one -> a real namespace. */
static bool deep_chain_refused(const char *root, const char *leaf)
{
	char level[PATH_MAX], next[PATH_MAX], link[PATH_MAX], name[PATH_MAX];
	bool prepared = true;
	for (uint32_t index = 0; prepared && index < CHAIN_LEVELS; index++) {
		(void)snprintf(name, sizeof(name),
			       "virtual/chain/level%u/slaves", index);
		prepared = fixture_directory(root, name);
	}
	for (uint32_t index = 0; prepared && index < CHAIN_LEVELS; index++) {
		(void)snprintf(name, sizeof(name), "virtual/chain/level%u",
			       index);
		prepared = fixture_join(level, root, name) &&
			   fixture_join(link, level, "slaves/next");
		(void)snprintf(name, sizeof(name), "virtual/chain/level%u",
			       index + 1u);
		prepared = prepared && fixture_join(next, root, name) &&
			   symlink(index + 1u < CHAIN_LEVELS ? next : leaf,
				   link) == 0;
	}
	struct local_device_description chained = { 0 };
	return prepared && fixture_join(level, root, "virtual/chain/level0") &&
	       local_device_resolve(level, &chained) == STATUS_UNSUPPORTED;
}

static void topology_cases(void)
{
	char root[] = "/tmp/anemone-device-policy-XXXXXX";
	if (mkdtemp(root) == NULL) {
		expect(false, "topology fixture opens");
		return;
	}
	/* Controllers with a serial (same drive renumbered under another path), without one (two
	 * controllers of one PCI function) and with blank serials on two functions. */
	enum fixture_namespace {
		SERIAL_FIRST,
		SERIAL_SECOND,
		USB_DISK,
		SERIAL_RENUMBERED,
		UNSERIALED,
		UNSERIALED_RENUMBERED,
		BLANK_FIRST,
		BLANK_SECOND,
		TUNNELLED,
		FIXTURE_NAMESPACES
	};
	static const char *const namespaces[FIXTURE_NAMESPACES] = {
		"pci/0000:44:00.0/nvme/nvme0/nvme0n1",
		"pci/0000:42:00.0/nvme/nvme1/nvme1n1",
		"pci/usb1/block/sda",
		"moved/0000:44:00.0/nvme/nvme7/nvme7n2",
		"pci/0000:48:00.0/nvme/nvme2/nvme2n1",
		"pci/0000:48:00.0/nvme/nvme9/nvme9n2",
		"pci/0000:4a:00.0/nvme/nvme3/nvme3n1",
		"pci/0000:4c:00.0/nvme/nvme4/nvme4n1",
		"pci/0000:4e:00.0/0000:4f:00.0/nvme/nvme5/nvme5n1"
	};
	static const struct {
		const char *path, *name, *text;
	} attributes[] = {
		{ "pci/0000:44:00.0/nvme/nvme0", "serial", "SERIAL-A\n" },
		{ "pci/0000:44:00.0/nvme/nvme0", "model", "MODEL-X\n" },
		{ "pci/0000:42:00.0/nvme/nvme1", "serial", "SERIAL-B\n" },
		{ "pci/0000:42:00.0/nvme/nvme1", "model", "MODEL-X\n" },
		{ "moved/0000:44:00.0/nvme/nvme7", "serial", "SERIAL-A\n" },
		{ "moved/0000:44:00.0/nvme/nvme7", "model", "MODEL-X\n" },
		/* Namespace ids differ per namespace; they must not split one controller. */
		{ "pci/0000:48:00.0/nvme/nvme2/nvme2n1", "wwid", "eui.0001\n" },
		{ "pci/0000:48:00.0/nvme/nvme9/nvme9n2", "wwid", "eui.0002\n" },
		{ "pci/0000:4a:00.0/nvme/nvme3", "serial", "    \n" },
		{ "pci/0000:4a:00.0/nvme/nvme3", "model", "MODEL-X\n" },
		{ "pci/0000:4c:00.0/nvme/nvme4", "serial", "    \n" },
		{ "pci/0000:4c:00.0/nvme/nvme4", "model", "MODEL-X\n" },
		{ "pci/0000:4e:00.0/0000:4f:00.0/nvme/nvme5", "serial",
		  "SERIAL-T\n" },
		/* The kernel marks PCI devices below an external-facing (Thunderbolt/USB4) port removable. */
		{ "pci/0000:4e:00.0", "removable", "removable\n" },
		{ "pci/0000:42:00.0", "removable", "fixed\n" }
	};
	bool prepared = true;
	for (size_t index = 0; index < FIXTURE_NAMESPACES; index++) {
		char relative[PATH_MAX];
		(void)snprintf(relative, sizeof(relative), "%s/queue",
			       namespaces[index]);
		prepared = fixture_directory(root, relative) && prepared;
		(void)snprintf(relative, sizeof(relative),
			       "%s/queue/rotational", namespaces[index]);
		prepared = fixture_file(root, relative, "0\n") && prepared;
		(void)snprintf(relative, sizeof(relative), "%s/removable",
			       namespaces[index]);
		prepared = fixture_file(root, relative, "0\n") && prepared;
	}
	for (size_t index = 0;
	     index < sizeof(attributes) / sizeof(attributes[0]); index++) {
		char relative[PATH_MAX];
		(void)snprintf(relative, sizeof(relative), "%s/%s",
			       attributes[index].path, attributes[index].name);
		prepared =
			fixture_file(root, relative, attributes[index].text) &&
			prepared;
	}
	prepared = fixture_directory(
			   root,
			   "pci/0000:44:00.0/nvme/nvme0/nvme0n1/nvme0n1p1") &&
		   prepared;
	prepared =
		fixture_file(
			root,
			"pci/0000:44:00.0/nvme/nvme0/nvme0n1/nvme0n1p1/partition",
			"1\n") &&
		prepared;
	prepared = fixture_directory(root, "virtual/block/dm-0/slaves") &&
		   prepared;
	char first[PATH_MAX], mapped[PATH_MAX], link[PATH_MAX];
	char resolved[FIXTURE_NAMESPACES][PATH_MAX];
	for (size_t index = 0; index < FIXTURE_NAMESPACES; index++)
		prepared = fixture_join(resolved[index], root,
					namespaces[index]) &&
			   prepared;
	prepared = fixture_join(first, resolved[SERIAL_FIRST], "nvme0n1p1") &&
		   fixture_join(mapped, root, "virtual/block/dm-0") &&
		   fixture_join(link, mapped, "slaves/first") &&
		   symlink(first, link) == 0 &&
		   fixture_join(link, mapped, "slaves/second") &&
		   symlink(resolved[SERIAL_SECOND], link) == 0 && prepared;
	expect(prepared,
	       "bounded dm/md/partition/USB topology fixture is prepared");
	if (prepared) {
		struct local_device_description partition = { 0 },
						volume = { 0 },
						external = { 0 };
		/* About 64 KiB each: kept off the stack. */
		static struct local_device_description
			devices[FIXTURE_NAMESPACES];
		bool resolved_all = true;
		for (size_t index = 0; index < FIXTURE_NAMESPACES; index++)
			resolved_all = local_device_resolve(resolved[index],
							    &devices[index]) ==
					       STATUS_OK &&
				       devices[index].count == 1 &&
				       resolved_all;
		expect(resolved_all, "every fixture namespace resolves");
		expect(local_device_resolve(first, &partition) == STATUS_OK &&
			       partition.count == 1,
		       "partition resolves to one physical controller");
		expect(local_device_overlaps(&partition,
					     &devices[SERIAL_RENUMBERED]) &&
			       !local_device_overlaps(
				       &devices[SERIAL_SECOND],
				       &devices[SERIAL_RENUMBERED]),
		       "controller renumbering and multiple namespaces retain the same physical identity");
		expect(local_device_overlaps(&devices[UNSERIALED],
					     &devices[UNSERIALED_RENUMBERED]) &&
			       !local_device_overlaps(&devices[UNSERIALED],
						      &partition),
		       "without a serial the PCI function keeps renumbered controllers and namespaces together");
		expect(!local_device_overlaps(&devices[BLANK_FIRST],
					      &devices[BLANK_SECOND]) &&
			       !local_device_overlaps(&devices[BLANK_FIRST],
						      &devices[UNSERIALED]),
		       "blank serials do not merge distinct drives");
		expect(local_device_resolve(mapped, &volume) == STATUS_OK &&
			       volume.count == TEST_SECOND_DEVICE &&
			       local_device_overlaps(&partition, &volume),
		       "dm/md slaves resolve to the physical overlap set");
		expect(local_device_resolve(resolved[USB_DISK], &external) ==
				       STATUS_OK &&
			       external.external &&
			       local_device_media_refusal(&external, false) !=
				       NULL &&
			       local_device_media_refusal(&external, true) ==
				       NULL,
		       "USB sysfs ancestry is excluded by default and accepted with override");
		expect(devices[TUNNELLED].external &&
			       local_device_media_refusal(&devices[TUNNELLED],
							  false) != NULL &&
			       local_device_media_refusal(&devices[TUNNELLED],
							  true) == NULL &&
			       !devices[SERIAL_SECOND].external,
		       "a Thunderbolt/USB4-tunnelled NVMe is external: only the override admits it; fixed stays "
		       "internal");
		expect(deep_chain_refused(root, resolved[SERIAL_FIRST]),
		       "a slave chain deeper than the bound is refused without recursion");
	}
	expect(local_device_external_bus("usb") &&
		       local_device_external_bus("mmc") &&
		       !local_device_external_bus("ata") &&
		       !local_device_external_bus(NULL),
	       "udev ID_BUS external classification");
	expect(fixture_remove(root),
	       "owned topology fixture removed without following its symlinks");
}

static void policy_cases(void)
{
	struct local_device_description first = {
		.count = 1,
		.physical = { 1 },
		.total_bytes = TEST_TOTAL_BYTES,
		.available_bytes = TEST_AVAILABLE_BYTES
	};
	expect(local_device_media_refusal(&first, false) == NULL,
	       "internal SSD admitted");
	first.external = true;
	expect(strcmp(local_device_media_refusal(&first, false),
		      "external_device_requires_override") == 0,
	       "USB/removable default exclusion");
	expect(local_device_media_refusal(&first, true) == NULL,
	       "external knob admits removable device");
	first.network = true;
	expect(strcmp(local_device_media_refusal(&first, true),
		      "network_filesystem_or_transport") == 0,
	       "external knob never admits network");
	static const char *const network_names[] = {
		"nfs",	      "nfs4", "cifs", "smb3",	  "sshfs",
		"fuse.sshfs", "9p",   "ceph", "glusterfs"
	};
	for (size_t index = 0;
	     index < sizeof(network_names) / sizeof(network_names[0]); index++)
		expect(local_device_network_filesystem(network_names[index]),
		       "network mount is refused before stat or I/O");
	expect(!local_device_network_filesystem("ext4"),
	       "local filesystem remains eligible for topology checks");
	first.network = false;
	first.virtual_media = true;
	expect(strcmp(local_device_media_refusal(&first, true),
		      "virtual_filesystem_or_device") == 0,
	       "virtual filesystems and loop images never admitted");
	first.virtual_media = false;
	first.external = false;
	first.rotational = true;
	expect(strcmp(local_device_media_refusal(&first, false),
		      "rotational_device") == 0 &&
		       strcmp(local_device_media_refusal(&first, true),
			      "rotational_device") == 0,
	       "internal rotational media stay excluded");
	expect(local_device_space_refusal(
		       TEST_TOTAL_BYTES, TEST_AVAILABLE_BYTES, TEST_COPY_BYTES,
		       LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT, 0, false) == NULL,
	       "copy plus exact floor fits");
	expect(strcmp(local_device_space_refusal(
			      TEST_TOTAL_BYTES, TEST_AVAILABLE_BYTES - 1u,
			      TEST_COPY_BYTES,
			      LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT, 0, false),
		      "insufficient_space") == 0,
	       "insufficient space excluded before copying");
	expect(local_device_space_refusal(
		       TEST_TOTAL_BYTES, TEST_AVAILABLE_BYTES - 1u,
		       TEST_COPY_BYTES, TEST_CUSTOM_FLOOR, 0, false) == NULL,
	       "configurable floor used");
	expect(strcmp(local_device_space_refusal(
			      TEST_TOTAL_BYTES, 1, 0,
			      LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT, 0, false),
		      "insufficient_space") == 0,
	       "publication check accounts for already allocated copy");
	expect(local_device_space_refusal(UINT64_MAX, UINT64_MAX, 0,
					  LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT, 0,
					  false) == NULL,
	       "floor arithmetic bounded at uint64 limit");
	struct local_device_description alias = { .count = 1,
						  .physical = { 1 } };
	expect(local_device_overlaps(&first, &alias),
	       "two mounts or partitions of same disk counted once");
	first.count = TEST_SECOND_DEVICE;
	first.physical[1] = TEST_SECOND_DEVICE;
	alias.count = TEST_SECOND_DEVICE;
	alias.physical[0] = TEST_SECOND_DEVICE;
	alias.physical[1] = TEST_THIRD_DEVICE;
	expect(local_device_overlaps(&first, &alias),
	       "overlapping dm/md physical sets counted once");
}

/* Stacked entries on one point: the last listed (the visible nfs4 over its autofs) names it, so describe returns
 * network before any stat/statfs. Points stay raw (the path a replica directory is built from). */
static void mount_cases(void)
{
	char table[] = "systemd-1 /mnt/nas autofs rw,relatime 0 0\n"
		       "nas:/export /mnt/nas nfs4 rw,relatime 0 0\n"
		       "/dev/nvme0n1p1 /mnt/Donn\xc3\xa9"
		       "es ext4 rw 0 0\n";
	FILE *const mounts = fmemopen(table, sizeof(table) - 1u, "r");
	expect(mounts != NULL, "mount table fixture opens");
	if (mounts == NULL)
		return;
	struct local_device_description stacked = { 0 }, raw = { 0 };
	local_device_mount_of(mounts, "/mnt/nas/model.anemone", &stacked);
	expect(strcmp(stacked.mount, "/mnt/nas") == 0 &&
		       strcmp(stacked.filesystem, "nfs4") == 0 &&
		       local_device_network_filesystem(stacked.filesystem),
	       "the visible stacked mount names a network path before any filesystem call");
	expect(fseek(mounts, 0L, SEEK_SET) == 0, "the mount table rewinds");
	clearerr(mounts);
	local_device_mount_of(mounts,
			      "/mnt/Donn\xc3\xa9"
			      "es/anemone_engram_rnd",
			      &raw);
	expect(strcmp(raw.mount, "/mnt/Donn\xc3\xa9"
				 "es") == 0 &&
		       strcmp(raw.filesystem, "ext4") == 0,
	       "a non-ASCII mount point stays the raw path for replica directories");
	(void)fclose(mounts);
}

static void printed_cases(void)
{
	struct local_device_description device = { .count = 1,
						   .physical = { 1 },
						   .available_bytes =
							   TEST_SPACE_AVAILABLE,
						   .mount = "/mnt/test",
						   .model = "fixture_ssd",
						   .transport = "usb",
						   .filesystem = "nfs4" };
	static const struct {
		enum local_device_reason reason;
		const char *line;
	} cases[] = {
		{ REPLICA_SPACE,
		  "storage: replica /mnt/test (fixture_ssd, usb, 0000000000000001) excluded: insufficient_space "
		  "needs_bytes=517 floor_percent=20 available_bytes=310\n" },
		{ REPLICA_EXTERNAL,
		  "storage: replica /mnt/test (fixture_ssd, usb, 0000000000000001) excluded: "
		  "external_device_requires_override (use --replica_external=1)\n" },
		{ REPLICA_NETWORK,
		  "storage: replica /mnt/test (fixture_ssd, usb, 0000000000000001) excluded: "
		  "network_filesystem_or_transport filesystem=nfs4\n" },
		{ REPLICA_ROTATIONAL,
		  "storage: replica /mnt/test (fixture_ssd, usb, 0000000000000001) excluded: rotational_device\n" },
		{ REPLICA_DUPLICATE,
		  "storage: replica /mnt/test (fixture_ssd, usb, 0000000000000001) excluded: "
		  "duplicate_physical_device\n" },
		{ REPLICA_DISAPPEARED,
		  "storage: replica /mnt/test (fixture_ssd, usb, 0000000000000001) excluded: "
		  "device_disappeared_or_unknown\n" },
		{ REPLICA_ID_CHANGED,
		  "storage: replica /mnt/test (fixture_ssd, usb, 0000000000000001) excluded: "
		  "hardware_identity_mismatch_retune_required\n" },
		{ REPLICA_VERIFIED,
		  "storage: replica /mnt/test (fixture_ssd, usb, 0000000000000001) included: "
		  "verified_identity_and_bytes load_weight=3 lookup_weight=7\n" }
	};
	for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]);
	     index++) {
		char *text = NULL;
		size_t bytes = 0;
		FILE *const stream = open_memstream(&text, &bytes);
		expect(stream != NULL, "product capture opens");
		if (stream == NULL)
			continue;
		local_device_print(
			stream, &device,
			local_device_reason_name(cases[index].reason),
			TEST_SPACE_BYTES, LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT,
			TEST_LOAD_WEIGHT, TEST_LOOKUP_WEIGHT);
		expect(fclose(stream) == 0 &&
			       strcmp(text, cases[index].line) == 0,
		       "exact release-visible product line for each candidate reason");
		free(text);
	}
	expect(!local_device_reason_known("untrusted_environment_text"),
	       "unknown reason never printed as supplied text");
	struct local_device_description kept = {
		.count = 1, .physical = { TEST_SECOND_DEVICE }
	};
	struct local_device_description duplicate = {
		.count = TEST_SECOND_DEVICE,
		.physical = { TEST_THIRD_DEVICE, TEST_SECOND_DEVICE }
	};
	char *text = NULL;
	size_t bytes = 0;
	FILE *const stream = open_memstream(&text, &bytes);
	expect(stream != NULL, "duplicate capture opens");
	if (stream == NULL)
		return;
	local_device_print_duplicate(stream, "/mnt/b/model\n.anemone",
				     &duplicate, "/mnt/a/model.anemone", &kept);
	local_device_print_duplicate(stream, "/mnt/c/model.anemone", &device,
				     "/mnt/a/model.anemone", &kept);
	expect(fclose(stream) == 0 &&
		       strcmp(text,
			      "storage: replica /mnt/b/model_.anemone excluded: duplicate_physical_device (shares "
			      "physical device 0000000000000002 with kept replica /mnt/a/model.anemone)\n") ==
			       0,
	       "duplicate line names both display-safe paths and the shared identity; distinct devices print nothing");
	free(text);
}

/* An idle automount: nothing is called below its point. Fired over an nfs4, the visible entry is the network one. */
static void automount_cases(void)
{
	char root[] = "/tmp/anemone-device-automount-XXXXXX", point[PATH_MAX],
	     path[PATH_MAX], table[PATH_MAX], lines[FIXTURE_TABLE_BYTES];
	if (mkdtemp(root) == NULL) {
		expect(false, "automount fixture opens");
		return;
	}
	const int length = snprintf(lines, sizeof(lines),
				    "systemd-1 %s/nas autofs rw,relatime 0 0\n",
				    root);
	bool prepared =
		length > 0 && (size_t)length < sizeof(lines) &&
		fixture_directory(root, "nas/anemone_engram_rnd/replicas") &&
		fixture_file(root, "mounts", lines) &&
		fixture_join(point, root, "nas") &&
		fixture_join(path, point, "anemone_engram_rnd/replicas") &&
		fixture_join(table, root, "mounts");
	struct local_device_description idle = { 0 }, fired = { 0 };
	expect(prepared && local_device_describe_mounts(table, path, &idle) ==
				   STATUS_UNSUPPORTED,
	       "a path below an automount that has not fired is refused before any stat or statfs");
	const int fired_length = snprintf(
		lines, sizeof(lines),
		"systemd-1 %s autofs rw,relatime 0 0\nnas:/export %s nfs4 rw 0 0\n",
		point, point);
	prepared = prepared && fired_length > 0 &&
		   (size_t)fired_length < sizeof(lines) &&
		   fixture_file(root, "mounts", lines);
	expect(prepared &&
		       local_device_describe_mounts(table, path, &fired) ==
			       STATUS_OK &&
		       fired.network && strcmp(fired.transport, "network") == 0,
	       "a fired automount over nfs4 is described as network without a filesystem call");
	expect(fixture_remove(root), "automount fixture removed");
}

int main(int argument_count, char **arguments)
{
	topology_cases();
	policy_cases();
	printed_cases();
	mount_cases();
	automount_cases();
	uint64_t value = UINT64_MAX;
	expect(local_device_identity(NULL, &value) == STATUS_INVALID_ARGUMENT &&
		       value == UINT64_MAX,
	       "invalid path leaves identity untouched");
	expect(local_device_identity("/proc/version", &value) ==
			       STATUS_UNSUPPORTED &&
		       value == UINT64_MAX,
	       "non-local filesystem refused");
	expect(local_device_read_probe("/proc/version", &value) ==
			       STATUS_UNSUPPORTED &&
		       value == UINT64_MAX,
	       "non-local probe refused without changing rate");
	for (int index = 1; index < argument_count; index++) {
		uint64_t identity = 0, rate = 0;
		const result_status media =
			local_device_identity(arguments[index], &identity);
		const result_status speed =
			local_device_read_probe(arguments[index], &rate);
		expect(media == STATUS_OK && speed == STATUS_OK &&
			       rate >= LOCAL_DEVICE_MINIMUM_MIB_PER_SECOND,
		       "existing candidate passes media and sampled speed admission");
		(void)printf(
			"device_identity=%016" PRIx64
			" media_status=%d probe_status=%d MiB_per_second=%" PRIu64
			" copy_room_status=%d\n",
			identity, (int)media, (int)speed, rate,
			(int)local_device_copy_room(arguments[index], 0));
	}
	return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
