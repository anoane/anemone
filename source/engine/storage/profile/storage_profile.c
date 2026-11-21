/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "storage_profile.h"
#include "engine/storage/io/local_device.h"
#include "engine/storage/load/loader.h"
#include "engine/storage/io/storage_exl3.h"
#include "engine/storage/load/tp_load.h"
#include "engine/engram/engram_step.h"
#include "common/json.h"
#include "common/sha256.h"
#include "engine/storage/replicas/storage_prepare.h"
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <mntent.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#define PROFILE_SCHEMA 2u
#define MANIFEST_SCHEMA 1u
#define MOST_DECISIONS 64u
#define DOCUMENT_BYTES (4u << 20u)
#define HASH_BUFFER_BYTES (1u << 20u)
#define HEX_CHARACTERS 2u
#define HEX_DIGITS 16u
#define KEY_INPUT_BYTES (32u << 10u)
#define MOST_ANCESTORS 24u
#define MOST_UUIDS 256u
#define MOST_CHUNKS 16384u
#define MOST_CHUNK_BYTES (64u << 20u)
#define MAXIMUM_WEIGHT 1000000u
#define MAXIMUM_THREADS LOADER_MAXIMUM_READER_THREADS_PER_COPY
#define MAXIMUM_CACHE_MIB 4096u
#define PERCENT 100u
#define SPLIT_NAME_BYTES \
	16u /* the longest copy split's name (tp_load_split_names) and its terminator */
#define VERIFY_SECONDS 1700
#define PARTIAL_SUFFIX ".partial"
#define DECIMAL 10
#define DEVICE_NUMBER_PARTS 2
#define CHUNK_ALIGNMENT_BYTES \
	4096u /* a manifest chunk: whole direct-I/O pages */
/* storage_profile_cli's argv after `anemone COMMAND`: its first, second and third arguments; a count equal to one
 * of them ends the arguments just before it */
enum storage_cli_argument { CLI_FIRST = 2, CLI_SECOND, CLI_THIRD };

struct document {
	char *bytes, *scratch_bytes;
	struct json_value root;
	struct json_scratch scratch;
};

static bool reason_set(char *reason, const char *text)
{
	(void)snprintf(reason, STORAGE_PROFILE_REASON_BYTES, "%s", text);
	return false;
}

static void document_close(struct document *document)
{
	free(document->bytes);
	free(document->scratch_bytes);
	memset(document, 0, sizeof(*document));
}

static bool document_open(const char *path, struct document *document)
{
	memset(document, 0, sizeof(*document));
	const int descriptor = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
	if (descriptor < 0)
		return false;
	struct stat file;
	bool valid = fstat(descriptor, &file) == 0 && S_ISREG(file.st_mode) &&
		     file.st_uid == geteuid() &&
		     (file.st_mode & (S_IWGRP | S_IWOTH)) == 0 &&
		     file.st_size > 0 &&
		     (uint64_t)file.st_size <= DOCUMENT_BYTES;
	if (valid) {
		document->bytes = malloc((size_t)file.st_size + 1);
		document->scratch_bytes = malloc(DOCUMENT_BYTES);
		valid = document->bytes != NULL &&
			document->scratch_bytes != NULL;
	}
	size_t done = 0;
	while (valid && done < (size_t)file.st_size) {
		const ssize_t count = read(descriptor, document->bytes + done,
					   (size_t)file.st_size - done);
		if (count < 0 && errno == EINTR)
			continue;
		valid = count > 0;
		if (valid)
			done += (size_t)count;
	}
	(void)close(descriptor);
	if (valid) {
		document->bytes[done] = '\0';
		document->scratch =
			(struct json_scratch){ .bytes = document->scratch_bytes,
					       .capacity = DOCUMENT_BYTES };
		valid = json_read_document(document->bytes, done,
					   &document->root) ==
				JSON_DOCUMENT_VALID &&
			document->bytes[document->root.begin] == '{' &&
			!json_has_repeated_key(&document->scratch,
					       document->root) &&
			!document->scratch.is_short;
	}
	if (!valid)
		document_close(document);
	return valid;
}

static struct json_value member(struct document *document,
				struct json_value object, const char *name)
{
	size_t cursor = object.begin;
	struct json_value key, value;
	if (object.text != NULL && object.text[object.begin] == '{')
		while (json_next(object, &cursor, &key, &value)) {
			document->scratch.used = 0;
			if (json_string_is(&document->scratch, key, name))
				return value;
		}
	return (struct json_value){ 0 };
}

static bool text_value(struct document *document, struct json_value value,
		       char *output, size_t capacity)
{
	if (value.text == NULL || value.text[value.begin] != '"')
		return false;
	document->scratch.used = 0;
	document->scratch.is_short = false;
	const struct json_bytes bytes =
		json_unescape(&document->scratch, value);
	if (document->scratch.is_short || bytes.count >= capacity ||
	    memchr(bytes.data, '\0', bytes.count) != NULL)
		return false;
	memcpy(output, bytes.data, bytes.count);
	output[bytes.count] = '\0';
	return true;
}

static bool number(struct json_value value, uint64_t most, uint64_t *output)
{
	if (value.text == NULL || value.begin == value.end)
		return false;
	uint64_t result = 0;
	for (size_t index = value.begin; index < value.end; index++) {
		const unsigned char byte = (unsigned char)value.text[index];
		if (byte < '0' || byte > '9' || result > most / DECIMAL ||
		    (result == most / DECIMAL &&
		     (uint64_t)(byte - '0') > most % DECIMAL))
			return false;
		result = result * DECIMAL + (uint64_t)(byte - '0');
	}
	*output = result;
	return true;
}

static bool field(struct document *document, struct json_value object,
		  const char *name, uint64_t most, uint32_t *output)
{
	uint64_t result = 0;
	if (!number(member(document, object, name), most, &result))
		return false;
	*output = (uint32_t)result;
	return true;
}

static bool hex_key(const char *key)
{
	if (strlen(key) != STORAGE_PROFILE_KEY_BYTES - 1)
		return false;
	return strspn(key, "0123456789abcdef") == STORAGE_PROFILE_KEY_BYTES - 1;
}

/* A digest as the profile writes it: lower-case hexadecimal, NUL-terminated. */
static void hash_text(const unsigned char digest[SHA256_DIGEST_BYTES],
		      char key[STORAGE_PROFILE_KEY_BYTES])
{
	static const char hex[] = "0123456789abcdef";
	for (uint32_t index = 0; index < SHA256_DIGEST_BYTES; index++) {
		key[(size_t)(index * HEX_CHARACTERS)] =
			hex[digest[index] / HEX_DIGITS];
		key[index * HEX_CHARACTERS + 1u] =
			hex[digest[index] % HEX_DIGITS];
	}
	key[STORAGE_PROFILE_KEY_BYTES - 1] = '\0';
}

static bool hash_memory(const void *bytes, size_t count, char *output)
{
	unsigned char digest[SHA256_DIGEST_BYTES];
	if (sha256_of(bytes, count, digest) != STATUS_OK)
		return false;
	hash_text(digest, output);
	return true;
}

static bool append_file(char *output, size_t *used, const char *directory,
			const char *name)
{
	char path[REPLICA_PATH_CAPACITY], text[REPLICA_PATH_CAPACITY];
	const int length =
		snprintf(path, sizeof(path), "%s/%s", directory, name);
	if (length < 0 || (size_t)length >= sizeof(path))
		return false;
	const int descriptor = open(path, O_RDONLY | O_CLOEXEC);
	if (descriptor < 0)
		return false;
	const ssize_t count = read(descriptor, text, sizeof(text) - 1);
	(void)close(descriptor);
	if (count <= 0 || (size_t)count >= sizeof(text) - 1)
		return false;
	text[count] = '\0';
	const int added = snprintf(output + *used, KEY_INPUT_BYTES - *used,
				   "%s=%s\n", name, text);
	if (added < 0 || (size_t)added >= KEY_INPUT_BYTES - *used)
		return false;
	*used += (size_t)added;
	return true;
}

#define UDEV_VALUE_BYTES 1024u
static bool append_udev(char *input, size_t *used, const char *device)
{
	char path[REPLICA_PATH_CAPACITY], number[REPLICA_PATH_CAPACITY];
	const int length = snprintf(path, sizeof(path), "%s/dev", device);
	if (length < 0 || (size_t)length >= sizeof(path))
		return false;
	FILE *file = fopen(path, "re");
	unsigned major_number = 0, minor_number = 0;
	const bool read = file != NULL &&
			  fscanf(file, "%u:%u", &major_number, &minor_number) ==
				  DEVICE_NUMBER_PARTS;
	if (file != NULL)
		(void)fclose(file);
	if (!read)
		return false;
	const int added = snprintf(path, sizeof(path), "/run/udev/data/b%u:%u",
				   major_number, minor_number);
	if (added < 0 || (size_t)added >= sizeof(path))
		return false;
	file = fopen(path, "re");
	if (file == NULL)
		return false;
	static const char *const names[] = {
		"E:ID_MODEL=", "E:ID_SERIAL=",	 "E:ID_SERIAL_SHORT=",
		"E:ID_WWN=",   "E:ID_REVISION=", "E:ID_BUS=",
		"E:ID_PATH="
	};
	char values[sizeof(names) / sizeof(names[0])][UDEV_VALUE_BYTES] = {
		{ 0 }
	};
	bool identity = false;
	for (uint32_t line = 0;
	     line < MOST_UUIDS && fgets(number, sizeof(number), file) != NULL;
	     line++) {
		for (size_t property = 0;
		     property < sizeof(names) / sizeof(names[0]); property++) {
			if (strncmp(number, names[property],
				    strlen(names[property])) != 0)
				continue;
			if (strlen(number) >= sizeof(values[property])) {
				(void)fclose(file);
				return false;
			}
			memcpy(values[property], number, strlen(number) + 1);
			identity = identity ||
				   strcmp(names[property], "E:ID_SERIAL=") ==
					   0 ||
				   strcmp(names[property], "E:ID_WWN=") == 0;
		}
	}
	(void)fclose(file);
	for (size_t property = 0; property < sizeof(names) / sizeof(names[0]);
	     property++) {
		const int amount = snprintf(input + *used,
					    KEY_INPUT_BYTES - *used, "%s",
					    values[property]);
		if (amount < 0 || (size_t)amount >= KEY_INPUT_BYTES - *used)
			return false;
		*used += (size_t)amount;
	}
	return identity;
}

result_status
storage_device_key_for(const char *path, bool external,
		       char key[STORAGE_PROFILE_KEY_BYTES],
		       struct local_device_description *description)
{
	if (path == NULL || key == NULL || description == NULL)
		return STATUS_INVALID_ARGUMENT;
	struct local_device_description device_info;
	const result_status status = local_device_describe(path, &device_info);
	if (status != STATUS_OK)
		return status;
	if (local_device_media_refusal(&device_info, external) != NULL)
		return STATUS_UNSUPPORTED;
	const uint64_t physical = local_device_group(&device_info);
	struct stat file;
	struct utsname kernel;
	char link[REPLICA_PATH_CAPACITY], input[KEY_INPUT_BYTES];
	if (stat(path, &file) != 0 || uname(&kernel) != 0)
		return STATUS_INPUT_OUTPUT_ERROR;
	const int prefix = snprintf(input, sizeof(input),
				    "kernel=%s\nphysical=%016" PRIx64
				    "\nexternal=%u\n",
				    kernel.release, physical,
				    device_info.external ? 1u : 0u);
	if (prefix < 0 || (size_t)prefix >= sizeof(input))
		return STATUS_UNSUPPORTED;
	size_t used = (size_t)prefix;
	/* Filesystem UUID binds partitions independently of device-number renumbering. */
	DIR *const uuids = opendir("/dev/disk/by-uuid");
	bool uuid_found = false;
	if (uuids != NULL) {
		struct dirent *entry;
		uint32_t count = 0;
		while (count++ < MOST_UUIDS &&
		       (entry = readdir(uuids)) != NULL) {
			struct stat block;
			const int length = snprintf(link, sizeof(link),
						    "/dev/disk/by-uuid/%s",
						    entry->d_name);
			if (length > 0 && (size_t)length < sizeof(link) &&
			    stat(link, &block) == 0 && S_ISBLK(block.st_mode) &&
			    block.st_rdev == file.st_dev) {
				const int added = snprintf(input + used,
							   sizeof(input) - used,
							   "uuid=%s\n",
							   entry->d_name);
				uuid_found = added > 0 &&
					     (size_t)added <
						     sizeof(input) - used;
				if (uuid_found)
					used += (size_t)added;
				break;
			}
		}
		(void)closedir(uuids);
	}
	FILE *const mounts = setmntent("/proc/self/mounts", "r");
	bool mount_found = false;
	if (mounts != NULL) {
		struct mntent record, *entry;
		char buffer[KEY_INPUT_BYTES];
		while ((entry = getmntent_r(mounts, &record, buffer,
					    sizeof(buffer))) != NULL) {
			struct stat mounted;
			if (strcmp(entry->mnt_dir, device_info.mount) == 0 &&
			    stat(entry->mnt_fsname, &mounted) == 0 &&
			    S_ISBLK(mounted.st_mode) &&
			    mounted.st_rdev == file.st_dev) {
				const int added = snprintf(
					input + used, sizeof(input) - used,
					"mount=%s\nfs=%s\noptions=%s\n",
					entry->mnt_dir, entry->mnt_type,
					entry->mnt_opts);
				mount_found = added > 0 &&
					      (size_t)added <
						      sizeof(input) - used;
				if (mount_found)
					used += (size_t)added;
				break;
			}
		}
		(void)endmntent(mounts);
	}

	for (uint32_t leaf = 0; leaf < device_info.count; leaf++) {
		char device[REPLICA_PATH_CAPACITY];
		memcpy(device, device_info.sysfs[leaf], sizeof(device));
		bool identified = append_udev(input, &used, device);
		identified = append_file(input, &used, device, "wwid") ||
			     identified;
		(void)append_file(input, &used, device,
				  "queue/logical_block_size");
		(void)append_file(input, &used, device,
				  "queue/physical_block_size");
		(void)append_file(input, &used, device, "device/model");
		identified =
			append_file(input, &used, device, "device/serial") ||
			identified;
		(void)append_file(input, &used, device, "device/rev");
		char *slash = strrchr(device, '/');
		if (slash != NULL &&
		    strncmp(slash + 1, "nvme", strlen("nvme")) == 0) {
			*slash = '\0';
			if (!append_file(input, &used, device, "model") ||
			    !append_file(input, &used, device, "serial") ||
			    !append_file(input, &used, device, "firmware_rev"))
				return STATUS_UNSUPPORTED;
			identified = true;
		}
		bool bus_found = false;
		for (uint32_t ancestor = 0; ancestor < MOST_ANCESTORS;
		     ancestor++) {
			const size_t saved = used;
			if (append_file(input, &used, device,
					"current_link_speed") &&
			    append_file(input, &used, device,
					"current_link_width") &&
			    append_file(input, &used, device,
					"max_link_speed") &&
			    append_file(input, &used, device,
					"max_link_width")) {
				const int address = snprintf(
					input + used, sizeof(input) - used,
					"pci=%s\n", device);
				if (address < 0 ||
				    (size_t)address >= sizeof(input) - used)
					return STATUS_UNSUPPORTED;
				used += (size_t)address;
				bus_found = true;
				break;
			}
			used = saved;
			if (append_file(input, &used, device, "idVendor") &&
			    append_file(input, &used, device, "idProduct") &&
			    append_file(input, &used, device, "bcdDevice") &&
			    append_file(input, &used, device, "speed") &&
			    append_file(input, &used, device, "devpath")) {
				char controller[REPLICA_PATH_CAPACITY];
				memcpy(controller, device, strlen(device) + 1);
				char *const usb = strstr(controller, "/usb");
				if (usb != NULL)
					*usb = '\0';
				const int address = snprintf(
					input + used, sizeof(input) - used,
					"usb_controller=%s\n", controller);
				if (address < 0 ||
				    (size_t)address >= sizeof(input) - used)
					return STATUS_UNSUPPORTED;
				used += (size_t)address;
				bus_found = true;
				break;
			}
			used = saved;
			if (append_file(input, &used, device, "cid") &&
			    append_file(input, &used, device, "manfid")) {
				(void)append_file(input, &used, device,
						  "fwrev");
				(void)append_file(input, &used, device,
						  "hwrev");
				identified = true;
				bus_found = true;
				break;
			}
			used = saved;
			slash = strrchr(device, '/');
			if (slash == NULL || slash == device)
				break;
			*slash = '\0';
		}
		if (!identified || !bus_found)
			return STATUS_UNSUPPORTED;
	}
	char result[STORAGE_PROFILE_KEY_BYTES];
	if (!uuid_found || !mount_found || !hash_memory(input, used, result))
		return STATUS_UNSUPPORTED;
	memcpy(key, result, sizeof(result));
	*description = device_info;
	return STATUS_OK;
}

result_status storage_device_key(const char *path,
				 char key[STORAGE_PROFILE_KEY_BYTES],
				 uint64_t *physical_out)
{
	if (physical_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	struct local_device_description description;
	const result_status status =
		storage_device_key_for(path, false, key, &description);
	if (status == STATUS_OK)
		*physical_out = local_device_group(&description);
	return status;
}

/* count bytes of a device from offset into buffer: a file's, or (map non-NULL) an EXL3 map's logical bytes through
 * its reader, in whole map blocks into a buffer aligned to one. */
static bool device_read(int descriptor, const struct storage_reader *map,
			uint64_t offset, unsigned char *buffer, size_t count)
{
	if (map != NULL)
		return storage_exl3.reader_read(map, 0, offset, buffer,
						count) == STATUS_OK;
	for (size_t done = 0; done < count;) {
		const ssize_t got = pread(descriptor, buffer + done,
					  count - done, (off_t)(offset + done));
		if (got < 0 && errno == EINTR)
			continue;
		if (got <= 0)
			return false;
		done += (size_t)got;
	}
	return true;
}

_Static_assert(sizeof(struct pack_header) <= EXL3_MAP_BLOCK_BYTES,
	       "a pack header lies in a map's first block");

/* Whether the device at path (its open descriptor, a regular file) holds bytes bytes: a file of that size, or an
 * EXL3 map (a profile's source) whose logical bytes are that many; the map's reader into map_out. */
static bool device_sized(const char *path, int descriptor, uint64_t bytes,
			 struct stat *file, struct storage_reader **map_out)
{
	struct storage_description logical = { 0 };
	const bool regular = fstat(descriptor, file) == 0 &&
			     S_ISREG(file->st_mode) && file->st_size > 0;
	if (storage_for_path(path) != &storage_exl3)
		return regular && (uint64_t)file->st_size == bytes;
	return regular &&
	       storage_exl3.reader_open(&path, 1, map_out) == STATUS_OK &&
	       storage_exl3.reader_describe(*map_out, &logical) == STATUS_OK &&
	       logical.byte_count == bytes;
}

/* buffer: HASH_BUFFER_BYTES aligned to an EXL3 map block. An EXL3 map (the source of an EXL3 profile) is verified by
 * its logical bytes, read through the map reader; its header is the logical pack's. */
static bool replica_verified(const char *path, struct document *manifest,
			     uint64_t bytes, uint64_t chunk_bytes,
			     unsigned char *buffer, time_t deadline,
			     const char *pack_identity)
{
	struct stat before, after;
	const int descriptor = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
	if (descriptor < 0)
		return false;
	struct storage_reader *map = NULL;
	bool valid = device_sized(path, descriptor, bytes, &before, &map);
	const struct json_value chunks =
		member(manifest, manifest->root, "chunks");
	size_t cursor = chunks.begin;
	uint64_t offset = 0;
	struct json_value value;
	while (valid && json_next(chunks, &cursor, NULL, &value)) {
		char expected[STORAGE_PROFILE_KEY_BYTES],
			actual[STORAGE_PROFILE_KEY_BYTES];
		valid = offset < bytes &&
			text_value(manifest, value, expected,
				   sizeof(expected)) &&
			hex_key(expected);
		const uint64_t amount = bytes - offset < chunk_bytes ?
						bytes - offset :
						chunk_bytes;
		struct sha256_stream stream;
		sha256_begin(&stream);
		uint64_t done = 0;
		while (valid && done < amount) {
			const uint64_t remaining = amount - done;
			const size_t wanted = remaining < HASH_BUFFER_BYTES ?
						      (size_t)remaining :
						      HASH_BUFFER_BYTES;
			valid = time(NULL) <= deadline &&
				device_read(descriptor, map, offset + done,
					    buffer, wanted) &&
				sha256_feed(&stream, buffer, wanted) ==
					STATUS_OK;
			done += wanted;
		}
		unsigned char digest[SHA256_DIGEST_BYTES];
		if (valid)
			valid = sha256_finish(&stream, digest) == STATUS_OK;
		if (valid) {
			hash_text(digest, actual);
			valid = strcmp(actual, expected) == 0;
		}
		offset += amount;
	}
	valid = valid && offset == bytes && fstat(descriptor, &after) == 0 &&
		before.st_size == after.st_size &&
		before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
		before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
		before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
		before.st_ctim.tv_nsec == after.st_ctim.tv_nsec;
	struct pack_header header = { 0 };
	char identity[STORAGE_PROFILE_KEY_BYTES];
	if (valid) {
		valid = device_read(descriptor, map, 0, buffer,
				    map != NULL ? EXL3_MAP_BLOCK_BYTES :
						  sizeof(header));
		if (valid)
			memcpy(&header, buffer, sizeof(header));
		valid = valid && header.file_byte_count == bytes;
		if (valid) {
			(void)snprintf(identity, sizeof(identity),
				       "%016" PRIx64, header.header_digest);
			valid = strcmp(identity, pack_identity) == 0;
		}
	}
	storage_exl3.reader_close(map);
	(void)close(descriptor);
	return valid;
}

bool storage_pack_header_read(const char *path, struct pack_header *header_out,
			      uint64_t *bytes_out)
{
	_Alignas(
		EXL3_MAP_BLOCK_BYTES) unsigned char block[EXL3_MAP_BLOCK_BYTES];
	struct stat file;
	struct storage_reader *map = NULL;
	struct storage_description logical = { 0 };
	const bool is_map = storage_for_path(path) == &storage_exl3;
	const int descriptor =
		path != NULL ? open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW |
						  O_NONBLOCK) :
			       -1;
	if (descriptor < 0)
		return false;
	const bool valid =
		fstat(descriptor, &file) == 0 && S_ISREG(file.st_mode) &&
		file.st_size > 0 &&
		(!is_map ||
		 (storage_exl3.reader_open(&path, 1, &map) == STATUS_OK &&
		  storage_exl3.reader_describe(map, &logical) == STATUS_OK)) &&
		device_read(descriptor, map, 0, block,
			    is_map ? sizeof(block) : sizeof(*header_out));
	storage_exl3.reader_close(map);
	(void)close(descriptor);
	if (valid) {
		memcpy(header_out, block, sizeof(*header_out));
		*bytes_out = is_map ? logical.byte_count :
				      (uint64_t)file.st_size;
	}
	return valid;
}

/* The source is the profile's pack: its header digest and byte count, read as the loader reads them. */
static bool source_identity_matches(const struct storage_profile *profile)
{
	struct pack_header header = { 0 };
	uint64_t bytes = 0;
	char identity[STORAGE_PROFILE_KEY_BYTES];
	if (!storage_pack_header_read(profile->source, &header, &bytes))
		return false;
	(void)snprintf(identity, sizeof(identity), "%016" PRIx64,
		       header.header_digest);
	return bytes == profile->source_bytes &&
	       header.file_byte_count == profile->source_bytes &&
	       strcmp(identity, profile->pack_identity) == 0;
}

static bool saved_device(struct document *document, struct json_value decision,
			 struct local_device_description *device)
{
	memset(device, 0, sizeof(*device));
	if (!text_value(document, member(document, decision, "mount"),
			device->mount, sizeof(device->mount)) ||
	    !text_value(document, member(document, decision, "model"),
			device->model, sizeof(device->model)) ||
	    !text_value(document, member(document, decision, "transport"),
			device->transport, sizeof(device->transport)) ||
	    !text_value(document, member(document, decision, "filesystem"),
			device->filesystem, sizeof(device->filesystem)))
		return false;
	const struct json_value physical =
		member(document, decision, "physical_devices");
	if (physical.text == NULL || physical.text[physical.begin] != '[')
		return false;
	size_t cursor = physical.begin;
	struct json_value value;
	while (json_next(physical, &cursor, NULL, &value)) {
		if (device->count == LOCAL_DEVICE_PHYSICAL_MOST)
			return false;
		char text[STORAGE_PROFILE_KEY_BYTES];
		if (!text_value(document, value, text, sizeof(text)) ||
		    strlen(text) != HEX_DIGITS ||
		    strspn(text, "0123456789abcdef") != HEX_DIGITS)
			return false;
		char *end = NULL;
		errno = 0;
		const unsigned long long parsed =
			strtoull(text, &end, HEX_DIGITS);
		if (errno != 0 || *end != '\0')
			return false;
		device->physical[device->count++] = (uint64_t)parsed;
	}
	return true;
}

/* The reason for a path local_device_describe refused: below an automount that has not fired (its guard names the
 * autofs entry and touches nothing), automount_not_mounted, as autotune records it; otherwise the device is gone. */
static const char *
describe_refusal(const struct local_device_description *described)
{
	return strcmp(described->filesystem, "autofs") == 0 ?
		       "automount_not_mounted" :
		       "device_disappeared_or_unknown";
}

static void decision_changed(const char *saved, const char *current)
{
	if (local_device_reason_known(saved) &&
	    local_device_reason_known(current) && strcmp(saved, current) != 0)
		(void)fprintf(stderr,
			      "storage: replica decision changed: %s -> %s\n",
			      saved, current);
}

static void observe(struct storage_profile *profile,
		    const struct local_device_description *device,
		    const char *reason, uint32_t load_weight,
		    uint32_t lookup_weight, bool product)
{
	if (profile->observation_count == STORAGE_PROFILE_OBSERVATIONS)
		return;
	struct storage_replica_observation *const entry =
		&profile->observations[profile->observation_count++];
	(void)snprintf(entry->mount, sizeof(entry->mount), "%s", device->mount);
	(void)snprintf(entry->reason, sizeof(entry->reason), "%s",
		       local_device_reason_known(reason) ? reason :
							   "invalid_decision");
	entry->included =
		strcmp(entry->reason, "verified_identity_and_bytes") == 0;
	entry->load_reason[0] = entry->lookup_reason[0] = '\0';
	entry->load_weight = load_weight;
	entry->lookup_weight = lookup_weight;
	entry->load_active = entry->included && profile->load_enabled != 0 &&
			     load_weight != 0;
	entry->lookup_active = entry->included &&
			       profile->engram_enabled != 0 &&
			       lookup_weight != 0;
	entry->physical_count = device->count;
	memcpy(entry->physical, device->physical, sizeof(entry->physical));
	if (product)
		local_device_print(
			stderr, device, entry->reason, 0,
			profile->replica_floor_percent,
			profile->load_enabled != 0 ? load_weight : 0,
			profile->engram_enabled != 0 ? lookup_weight : 0);
}

static struct storage_profile last_decisions;

const struct storage_profile *storage_profile_decisions(void)
{
	return &last_decisions;
}

uint32_t storage_profile_load_list(const struct storage_profile *profile,
				   const char *paths[STORAGE_MAXIMUM_COPIES],
				   uint32_t weights[STORAGE_MAXIMUM_COPIES])
{
	uint32_t count = 0;
	/* A zero load share keeps a device for lookup only; the loader refuses zero shares in a tuned list. */
	for (uint32_t index = 0; index < profile->count; index++)
		if (profile->load_weights[index] != 0) {
			paths[count] = profile->paths[index];
			weights[count++] = profile->load_weights[index];
		}
	return count;
}

result_status
storage_profile_note_loaded(FILE *output,
			    const struct local_device_description *device,
			    const char *refusal, uint32_t floor_percent)
{
	/* The loader reads this copy whatever the media policy says: it is included, the refusal only advisory. */
	local_device_print(output, device, "eligible", 0, floor_percent, 0, 0);
	if (refusal != NULL && local_device_reason_known(refusal)) {
		char shown[REPLICA_PATH_CAPACITY];
		local_device_display(shown, sizeof(shown), device->mount);
		(void)fprintf(
			output,
			"storage: loaded copy %s advisory: %s (given copies load as given)\n",
			shown[0] != '\0' ? shown : "unavailable", refusal);
	}
	return storage_profile_note(device, "eligible", STORAGE_REPLICA_LOAD);
}

bool storage_profile_engram_argument(const char *argument)
{
	/* --engram_shared (a TP host-gather staging knob) is not storage lookup and keeps the tuned profile. */
	static const char *const family[] = {
		"--engram_io=", "--engram_ram_percent=", "--engram_cache_mib=",
		"--engram_sources=", "--turbo-nvme-engrams="
	};
	for (size_t index = 0; index < sizeof(family) / sizeof(family[0]);
	     index++)
		if (strncmp(argument, family[index], strlen(family[index])) ==
		    0)
			return true;
	return false;
}

bool storage_profile_engram_sources(const struct storage_profile *profile,
				    char *output, size_t capacity,
				    uint32_t *count_out)
{
	size_t used = 0;
	uint32_t count = 0;
	if (capacity == 0)
		return false;
	output[0] = '\0';
	for (uint32_t index = 0; index < profile->count; index++) {
		if (storage_for_path(profile->paths[index]) == &storage_exl3)
			continue;
		const int length = snprintf(output + used, capacity - used,
					    "%s%s", count != 0 ? "," : "",
					    profile->paths[index]);
		if (length < 0 || (size_t)length >= capacity - used) {
			output[0] = '\0';
			return false;
		}
		used += (size_t)length;
		count++;
	}
	*count_out = count;
	return true;
}

void storage_profile_roles(bool tuned_load, bool tuned_lookup, bool disk_lookup)
{
	for (uint32_t index = 0; index < last_decisions.observation_count;
	     index++) {
		if (!tuned_load)
			last_decisions.observations[index].load_active = false;
		if (!tuned_lookup)
			last_decisions.observations[index].lookup_active =
				false;
	}
	last_decisions.load_enabled = 1;
	last_decisions.engram_enabled = disk_lookup ? 1u : 0u;
}

result_status
storage_profile_note(const struct local_device_description *device,
		     const char *reason, enum storage_replica_role role)
{
	if (device == NULL ||
	    (role != STORAGE_REPLICA_LOAD && role != STORAGE_REPLICA_LOOKUP) ||
	    !local_device_reason_known(reason))
		return STATUS_INVALID_ARGUMENT;
	const bool included = strcmp(reason, "eligible") == 0 ||
			      strcmp(reason, "verified_identity_and_bytes") ==
				      0;
	uint32_t selected = last_decisions.observation_count;
	for (uint32_t index = 0; index < last_decisions.observation_count;
	     index++) {
		const struct storage_replica_observation *const entry =
			&last_decisions.observations[index];
		if (strcmp(entry->mount, device->mount) == 0) {
			selected = index;
			break;
		}
	}
	if (selected == last_decisions.observation_count) {
		if (selected == STORAGE_PROFILE_OBSERVATIONS)
			return STATUS_OK; /* diagnostics only: a full table drops this entry, never fails startup */
		observe(&last_decisions, device, reason, 0, 0, false);
	}
	struct storage_replica_observation *const entry =
		&last_decisions.observations[selected];
	char *const role_reason = role == STORAGE_REPLICA_LOAD ?
					  entry->load_reason :
					  entry->lookup_reason;
	(void)snprintf(role_reason, sizeof(entry->load_reason), "%s",
		       included ? "" : reason);
	if (!included) {
		if (role == STORAGE_REPLICA_LOAD)
			entry->load_active = false;
		else
			entry->lookup_active = false;
		if (!entry->load_active && !entry->lookup_active) {
			entry->included = false;
			(void)snprintf(entry->reason, sizeof(entry->reason),
				       "%s", reason);
		}
	}
	if (included) {
		entry->included = true;
		(void)snprintf(entry->reason, sizeof(entry->reason), "%s",
			       reason);
		if (role == STORAGE_REPLICA_LOAD)
			entry->load_active = true;
		else
			entry->lookup_active = true;
	}
	return STATUS_OK;
}

bool storage_profile_read(const char *path, struct storage_profile *output,
			  char reason[STORAGE_PROFILE_REASON_BYTES])
{
	if (path == NULL || output == NULL || reason == NULL)
		return false;
	memset(&last_decisions, 0, sizeof(last_decisions));
	struct document profile, manifest;
	if (!document_open(path, &profile))
		return reason_set(reason, access(path, F_OK) == 0 ?
						  "unreadable_or_invalid_json" :
						  "missing");
	struct storage_profile result = { 0 };
	uint32_t schema = 0, enabled = 0;
	bool valid =
		field(&profile, profile.root, "schema", UINT32_MAX, &schema) &&
		schema == PROFILE_SCHEMA;
	const char *failure = "schema_mismatch";
	if (valid) {
		valid = field(&profile, profile.root, "enabled", 1, &enabled) &&
			enabled == 1;
		failure = "disabled_or_unqualified";
	}
	char manifest_path[REPLICA_PATH_CAPACITY],
		expected[STORAGE_PROFILE_KEY_BYTES],
		actual[STORAGE_PROFILE_KEY_BYTES];
	if (valid) {
		valid = text_value(&profile,
				   member(&profile, profile.root, "source"),
				   result.source, sizeof(result.source)) &&
			text_value(&profile,
				   member(&profile, profile.root, "manifest"),
				   manifest_path, sizeof(manifest_path)) &&
			text_value(&profile,
				   member(&profile, profile.root,
					  "manifest_sha256"),
				   expected, sizeof(expected)) &&
			hex_key(expected) &&
			number(member(&profile, profile.root, "created_unix"),
			       INT64_MAX, &result.created_unix) &&
			result.created_unix <= (uint64_t)time(NULL) &&
			result.source[0] == '/' && manifest_path[0] == '/';
		failure = "invalid_profile_fields";
	}
	const struct json_value policy =
		member(&profile, profile.root, "replica_policy");
	if (valid) {
		valid = field(&profile, policy, "external", 1,
			      &result.replica_external) &&
			field(&profile, policy, "floor_percent", PERCENT - 1u,
			      &result.replica_floor_percent);
		failure = "invalid_replica_policy";
	}
	const struct json_value decisions =
		member(&profile, profile.root, "decisions");
	uint32_t decision_count = 0, included_decisions = 0;
	if (valid) {
		valid = decisions.text != NULL &&
			decisions.text[decisions.begin] == '[';
		size_t decision_cursor = decisions.begin;
		struct json_value decision;
		while (valid && json_next(decisions, &decision_cursor, NULL,
					  &decision)) {
			char saved_path[REPLICA_PATH_CAPACITY],
				saved_reason[STORAGE_PROFILE_REASON_BYTES];
			uint32_t included = 0;
			valid = decision_count++ < MOST_DECISIONS &&
				text_value(&profile,
					   member(&profile, decision, "path"),
					   saved_path, sizeof(saved_path)) &&
				saved_path[0] == '/' &&
				field(&profile, decision, "included", 1,
				      &included) &&
				text_value(&profile,
					   member(&profile, decision, "reason"),
					   saved_reason,
					   sizeof(saved_reason)) &&
				saved_reason[0] != '\0' &&
				local_device_reason_known(saved_reason);
			/* A repeated identical decision records the same replica twice; its duplicate device entry is
			   excluded below. A different decision for the same path contradicts it when either includes a
			   device; two excluded decisions (two refused mounts whose paths are no UTF-8 and
			   read alike, say) contradict nothing. */
			const size_t decision_bytes =
				decision.end - decision.begin;
			size_t earlier_cursor = decisions.begin;
			struct json_value earlier;
			while (valid &&
			       json_next(decisions, &earlier_cursor, NULL,
					 &earlier) &&
			       earlier.begin < decision.begin) {
				char earlier_path[REPLICA_PATH_CAPACITY];
				uint32_t earlier_included = 0;
				valid = text_value(&profile,
						   member(&profile, earlier,
							  "path"),
						   earlier_path,
						   sizeof(earlier_path)) &&
					field(&profile, earlier, "included", 1,
					      &earlier_included);
				valid = valid &&
					(strcmp(earlier_path, saved_path) !=
						 0 ||
					 (included == 0 &&
					  earlier_included == 0) ||
					 (earlier.end - earlier.begin ==
						  decision_bytes &&
					  memcmp(earlier.text + earlier.begin,
						 decision.text + decision.begin,
						 decision_bytes) == 0));
			}
			struct local_device_description recorded;
			valid = valid &&
				saved_device(&profile, decision, &recorded) &&
				(included == 0 ||
				 (recorded.mount[0] == '/' &&
				  recorded.count != 0 &&
				  (strcmp(saved_reason, "eligible") == 0 ||
				   strcmp(saved_reason,
					  "verified_identity_and_bytes") ==
					   0))) &&
				(included != 0 ||
				 (strcmp(saved_reason, "eligible") != 0 &&
				  strcmp(saved_reason,
					 "verified_identity_and_bytes") != 0));
			if (valid && included != 0)
				included_decisions++;
			/* An excluded network decision keeps its recorded reason: its path is never described again,
			   so startup never touches a network filesystem or fires its automount. */
			if (valid && included == 0 &&
			    strcmp(recorded.transport, "network") == 0) {
				result.excluded++;
				observe(&result, &recorded, saved_reason, 0, 0,
					true);
			} else if (valid && included == 0) {
				result.excluded++;
				struct local_device_description previous = { 0 },
								current = { 0 };
				previous = recorded;
				char parent[REPLICA_PATH_CAPACITY];
				memcpy(parent, saved_path,
				       strlen(saved_path) + 1);
				char *const slash = strrchr(parent, '/');
				if (slash != NULL)
					*(slash == parent ? slash + 1 : slash) =
						'\0';
				const result_status described =
					local_device_describe(parent, &current);
				const char *current_reason =
					described != STATUS_OK ?
						describe_refusal(&current) :
						local_device_media_refusal(
							&current,
							result.replica_external !=
								0);
				if (described == STATUS_OK)
					previous = current;
				if (current_reason == NULL)
					current_reason = local_device_space_refusal(
						current.total_bytes,
						current.available_bytes, 0,
						result.replica_floor_percent, 0,
						false);
				observe(&result, &previous,
					current_reason != NULL ?
						current_reason :
						"previously_excluded_retune_required",
					0, 0, true);
				decision_changed(
					saved_reason,
					current_reason != NULL ?
						current_reason :
						"previously_excluded_retune_required");
			}
		}
		failure = "invalid_decision_manifest";
	}
	const struct json_value load =
		member(&profile, profile.root, "turbo_load");
	const struct json_value engram =
		member(&profile, profile.root, "turbo_engram");
	if (valid) {
		char load_mechanism[STORAGE_PROFILE_REASON_BYTES],
			engram_mechanism[STORAGE_PROFILE_REASON_BYTES];
		valid = field(&profile, load, "enabled", 1,
			      &result.load_enabled) &&
			field(&profile, load, "threads", MAXIMUM_THREADS,
			      &result.load_threads) &&
			result.load_threads != 0 &&
			field(&profile, engram, "enabled", 1,
			      &result.engram_enabled) &&
			field(&profile, engram, "ram_percent", PERCENT - 1,
			      &result.ram_percent) &&
			field(&profile, engram, "cache_mib", MAXIMUM_CACHE_MIB,
			      &result.cache_mib) &&
			field(&profile, engram, "io", 1, &result.engram_io) &&
			text_value(&profile,
				   member(&profile, load, "mechanism"),
				   load_mechanism, sizeof(load_mechanism)) &&
			strcmp(load_mechanism, "direct_pread") == 0 &&
			text_value(&profile,
				   member(&profile, engram, "mechanism"),
				   engram_mechanism,
				   sizeof(engram_mechanism)) &&
			strcmp(engram_mechanism, result.engram_io != 0 ?
							 "uring_fixed_direct" :
							 "direct_pread") == 0;
		failure = "unsupported_action_parameters";
	}
	bool manifest_open = valid && document_open(manifest_path, &manifest);
	if (valid) {
		valid = manifest_open &&
			hash_memory(manifest.bytes, strlen(manifest.bytes),
				    actual) &&
			strcmp(actual, expected) == 0;
		failure = "manifest_missing_or_digest_mismatch";
	}
	uint64_t bytes = 0, chunk_bytes = 0;
	if (valid) {
		valid = field(&manifest, manifest.root, "schema", UINT32_MAX,
			      &schema) &&
			schema == MANIFEST_SCHEMA &&
			number(member(&manifest, manifest.root, "bytes"),
			       INT64_MAX, &bytes) &&
			bytes != 0 &&
			text_value(&manifest,
				   member(&manifest, manifest.root, "pack_id"),
				   result.pack_identity,
				   sizeof(result.pack_identity)) &&
			result.pack_identity[0] != '\0' &&
			number(member(&manifest, manifest.root, "chunk_bytes"),
			       MOST_CHUNK_BYTES, &chunk_bytes) &&
			chunk_bytes >= CHUNK_ALIGNMENT_BYTES &&
			chunk_bytes % CHUNK_ALIGNMENT_BYTES == 0 &&
			(bytes - 1) / chunk_bytes < MOST_CHUNKS;
		const struct json_value chunks =
			member(&manifest, manifest.root, "chunks");
		valid = valid && chunks.text != NULL &&
			chunks.text[chunks.begin] == '[';
		failure = "invalid_source_manifest";
		char manifest_source[REPLICA_PATH_CAPACITY];
		valid = valid &&
			text_value(&manifest,
				   member(&manifest, manifest.root, "source"),
				   manifest_source, sizeof(manifest_source)) &&
			strcmp(manifest_source, result.source) == 0;
		result.source_bytes = bytes;
	}
	/* aligned for an EXL3 map's reads (replica_verified) */
	unsigned char *const buffer =
		valid ? aligned_alloc(EXL3_MAP_BLOCK_BYTES, HASH_BUFFER_BYTES) :
			NULL;
	if (valid && !source_identity_matches(&result)) {
		const int descriptor =
			open(result.source,
			     O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
		result.source_unavailable =
			descriptor < 0 && (errno == ENOENT || errno == ENODEV ||
					   errno == ENXIO || errno == EIO);
		if (descriptor >= 0)
			(void)close(descriptor);
		valid = result.source_unavailable;
		failure = "source_identity_mismatch";
	}
	if (valid && buffer == NULL) {
		valid = false;
		failure = "verification_memory";
	}
	const struct json_value devices =
		member(&profile, profile.root, "devices");
	size_t cursor = devices.begin;
	struct json_value device;
	uint32_t seen_count = 0;
	struct local_device_description seen[STORAGE_MAXIMUM_COPIES];
	const time_t deadline = time(NULL) + VERIFY_SECONDS;
	if (valid &&
	    (devices.text == NULL || devices.text[devices.begin] != '[')) {
		valid = false;
		failure = "invalid_devices";
	}
	while (valid && json_next(devices, &cursor, NULL, &device)) {
		char replica[REPLICA_PATH_CAPACITY],
			key[STORAGE_PROFILE_KEY_BYTES],
			found[STORAGE_PROFILE_KEY_BYTES];
		uint32_t load_weight = 0, engram_weight = 0;
		if (seen_count == STORAGE_MAXIMUM_COPIES ||
		    !text_value(&profile, member(&profile, device, "path"),
				replica, sizeof(replica)) ||
		    replica[0] != '/' || strchr(replica, ',') != NULL ||
		    (strlen(replica) >= sizeof(PARTIAL_SUFFIX) - 1 &&
		     strcmp(replica + strlen(replica) -
				    (sizeof(PARTIAL_SUFFIX) - 1),
			    PARTIAL_SUFFIX) == 0) ||
		    !text_value(&profile, member(&profile, device, "key"), key,
				sizeof(key)) ||
		    !hex_key(key) ||
		    !field(&profile, device, "load_weight", MAXIMUM_WEIGHT,
			   &load_weight) ||
		    !field(&profile, device, "engram_weight", MAXIMUM_WEIGHT,
			   &engram_weight)) {
			valid = false;
			failure = "invalid_device_parameters";
			break;
		}
		seen_count++;
		/* an EXL3 map's device is its largest source's, not the map file's */
		char bytes_path[EXL3_MAP_PATH_BYTES];
		const char *const device_path =
			storage_device_path(replica, bytes_path);
		char parent[REPLICA_PATH_CAPACITY];
		memcpy(parent, device_path, strlen(device_path) + 1);
		char *const slash = strrchr(parent, '/');
		if (slash != NULL)
			*(slash == parent ? slash + 1 : slash) = '\0';
		bool recorded = false;
		struct local_device_description previous = { 0 };
		size_t decision_cursor = decisions.begin;
		struct json_value decision;
		while (json_next(decisions, &decision_cursor, NULL,
				 &decision)) {
			char saved_path[REPLICA_PATH_CAPACITY],
				saved_key[STORAGE_PROFILE_KEY_BYTES];
			uint32_t included = 0;
			if (field(&profile, decision, "included", 1,
				  &included) &&
			    included != 0 &&
			    text_value(&profile,
				       member(&profile, decision, "path"),
				       saved_path, sizeof(saved_path)) &&
			    text_value(&profile,
				       member(&profile, decision, "key"),
				       saved_key, sizeof(saved_key)) &&
			    strcmp(saved_path, replica) == 0 &&
			    strcmp(saved_key, key) == 0) {
				recorded = true;
				if (!saved_device(&profile, decision,
						  &previous)) {
					recorded = false;
					break;
				}
			}
		}
		if (!recorded) {
			valid = false;
			failure = "unrecorded_device_decision";
			break;
		}
		struct local_device_description current = { 0 };
		const result_status described =
			slash != NULL ?
				local_device_describe(parent, &current) :
				STATUS_UNSUPPORTED;
		const char *excluded =
			described != STATUS_OK ?
				"device_disappeared_or_unknown" :
				local_device_media_refusal(
					&current, result.replica_external != 0);
		if (excluded == NULL &&
		    storage_device_key_for(parent, result.replica_external != 0,
					   found, &current) != STATUS_OK)
			excluded = "hardware_identity_unavailable";
		if (excluded == NULL && strcmp(found, key) != 0)
			excluded = "hardware_identity_mismatch_retune_required";
		if (excluded == NULL)
			excluded = local_device_space_refusal(
				current.total_bytes, current.available_bytes, 0,
				result.replica_floor_percent, 0, false);
		uint32_t kept = result.count;
		for (uint32_t index = 0;
		     excluded == NULL && index < result.count; index++)
			if (local_device_overlaps(&seen[index], &current)) {
				excluded = "duplicate_physical_device";
				kept = index;
			}
		if (excluded == NULL && load_weight == 0 && engram_weight == 0)
			excluded = "zero_tuned_weight";
		if (excluded == NULL &&
		    !replica_verified(replica, &manifest, bytes, chunk_bytes,
				      buffer, deadline, result.pack_identity))
			excluded = "replica_missing_short_or_unverified";
		observe(&result, described == STATUS_OK ? &current : &previous,
			excluded != NULL ? excluded :
					   "verified_identity_and_bytes",
			load_weight, engram_weight, true);
		if (excluded != NULL) {
			if (kept != result.count)
				local_device_print_duplicate(stderr, replica,
							     &current,
							     result.paths[kept],
							     &seen[kept]);
			decision_changed("verified_identity_and_bytes",
					 excluded);
			result.excluded++;
			continue;
		}
		seen[result.count] = current;
		memcpy(result.paths[result.count], replica,
		       strlen(replica) + 1);
		result.load_weights[result.count] = load_weight;
		result.engram_weights[result.count] = engram_weight;
		result.count++;
	}
	if (valid && seen_count != included_decisions) {
		valid = false;
		failure = "unrecorded_device_decision";
	}
	uint64_t load_total = 0, engram_total = 0;
	for (uint32_t index = 0; index < result.count; index++) {
		load_total += result.load_weights[index];
		engram_total += result.engram_weights[index];
	}
	if (load_total == 0)
		result.load_enabled = 0;
	if (engram_total == 0)
		result.engram_enabled = 0;
	if (valid && result.count == 0) {
		valid = false;
		failure = "no_verified_replicas";
	}
	if (valid)
		valid = hash_memory(profile.bytes, strlen(profile.bytes),
				    result.identity);
	free(buffer);
	if (manifest_open)
		document_close(&manifest);
	document_close(&profile);
	last_decisions = result;
	if (!valid) {
		for (uint32_t index = 0;
		     index < last_decisions.observation_count; index++)
			if (last_decisions.observations[index].included) {
				last_decisions.observations[index].included =
					false;
				last_decisions.observations[index].load_active =
					false;
				last_decisions.observations[index]
					.lookup_active = false;
				(void)snprintf(last_decisions
						       .observations[index]
						       .reason,
					       STORAGE_PROFILE_REASON_BYTES,
					       "%s", "invalid_decision");
			}
		return reason_set(reason, failure);
	}
	*output = result;
	(void)snprintf(reason, STORAGE_PROFILE_REASON_BYTES, "%s",
		       result.excluded != 0 ? "degraded_replicas_excluded" :
					      "valid");
	return true;
}

static void metrics_label(FILE *output, const char *text)
{
	(void)fputc('"', output);
	for (size_t index = 0;
	     index < REPLICA_PATH_CAPACITY && text[index] != '\0'; index++) {
		const unsigned char byte = (unsigned char)text[index];
		if (byte == '"' || byte == '\\')
			(void)fputc('\\', output);
		if (byte == '\n')
			(void)fputs("\\n", output);
		else
			(void)fputc(byte < ' ' || byte > '~' ? '_' : byte,
				    output);
	}
	(void)fputc('"', output);
}

result_status storage_profile_metrics(FILE *output,
				      const struct storage_profile *profile,
				      bool enabled)
{
	if (output == NULL || profile == NULL ||
	    profile->observation_count > STORAGE_PROFILE_OBSERVATIONS)
		return STATUS_INVALID_ARGUMENT;
	if (!enabled)
		return STATUS_OK;
	(void)fputs(
		"# HELP anemone_storage_replica_info Startup replica selection by physical device and role.\n"
		"# TYPE anemone_storage_replica_info gauge\n",
		output);
	uint32_t emitted[STORAGE_PROFILE_OBSERVATIONS], emitted_count = 0;
	/* Prefer the included representative over an excluded duplicate mount. */
	for (uint32_t pass = 0; pass <= 1u; pass++) {
		for (uint32_t index = 0; index < profile->observation_count;
		     index++) {
			const struct storage_replica_observation *const entry =
				&profile->observations[index];
			if ((pass == 0) != entry->included)
				continue;
			bool duplicate = false;
			for (uint32_t previous = 0; previous < emitted_count;
			     previous++) {
				const struct storage_replica_observation *const
					seen = &profile->observations
							[emitted[previous]];
				duplicate = duplicate ||
					    strcmp(entry->mount, seen->mount) ==
						    0;
				for (uint32_t first = 0;
				     first < entry->physical_count &&
				     first < LOCAL_DEVICE_PHYSICAL_MOST;
				     first++)
					for (uint32_t second = 0;
					     second < seen->physical_count &&
					     second <
						     LOCAL_DEVICE_PHYSICAL_MOST;
					     second++)
						duplicate =
							duplicate ||
							entry->physical[first] ==
								seen->physical
									[second];
			}
			if (duplicate)
				continue;
			emitted[emitted_count++] = index;
			for (uint32_t role = 0; role <= 1u; role++) {
				const bool active =
					entry->included &&
					(role == 0 ? entry->load_active :
						     entry->lookup_active);
				const char *const refused =
					role == 0 ? entry->load_reason :
						    entry->lookup_reason;
				const char *const reason =
					active ? (local_device_reason_known(
							  entry->reason) ?
							  entry->reason :
							  "invalid_decision") :
					!entry->included ?
						 (local_device_reason_known(
							  entry->reason) ?
							  entry->reason :
							  "invalid_decision") :
					local_device_reason_known(refused) ?
						 refused :
					(role == 0 ?
						 profile->load_enabled :
						 profile->engram_enabled) == 0 ?
						 "action_disabled" :
					(role == 0 ?
						 entry->load_weight :
						 entry->lookup_weight) == 0 ?
						 "zero_tuned_weight" :
						 "action_disabled";
				(void)fputs(
					"anemone_storage_replica_info{mount=",
					output);
				metrics_label(output, entry->mount);
				(void)fprintf(
					output,
					",role=\"%s\",status=\"%s\",reason=\"%s\"} 1\n",
					role == 0 ? "load" : "lookup",
					active ? "included" : "excluded",
					reason);
			}
		}
	}
	return ferror(output) ? STATUS_INPUT_OUTPUT_ERROR : STATUS_OK;
}

static void saved_report(const char *path)
{
	struct document document;
	if (!document_open(path, &document))
		return;
	(void)fputs(
		"storage: saved replica decisions (before live revalidation)\n",
		stderr);
	const struct json_value decisions =
		member(&document, document.root, "decisions");
	if (decisions.text != NULL && decisions.text[decisions.begin] == '[') {
		size_t cursor = decisions.begin;
		struct json_value decision;
		uint32_t count = 0;
		while (count++ < MOST_DECISIONS &&
		       json_next(decisions, &cursor, NULL, &decision)) {
			struct local_device_description device;
			char reason[STORAGE_PROFILE_REASON_BYTES],
				path_text[REPLICA_PATH_CAPACITY];
			uint32_t load = 0, lookup = 0;
			if (!saved_device(&document, decision, &device) ||
			    !text_value(&document,
					member(&document, decision, "reason"),
					reason, sizeof(reason)) ||
			    !local_device_reason_known(reason) ||
			    !text_value(&document,
					member(&document, decision, "path"),
					path_text, sizeof(path_text)))
				continue;
			const struct json_value devices =
				member(&document, document.root, "devices");
			size_t device_cursor = devices.begin;
			struct json_value item;
			if (devices.text != NULL &&
			    devices.text[devices.begin] == '[')
				while (json_next(devices, &device_cursor, NULL,
						 &item)) {
					char candidate[REPLICA_PATH_CAPACITY];
					if (text_value(&document,
						       member(&document, item,
							      "path"),
						       candidate,
						       sizeof(candidate)) &&
					    strcmp(candidate, path_text) == 0) {
						(void)field(&document, item,
							    "load_weight",
							    MAXIMUM_WEIGHT,
							    &load);
						(void)field(&document, item,
							    "engram_weight",
							    MAXIMUM_WEIGHT,
							    &lookup);
					}
				}
			local_device_print(stderr, &device, reason, 0,
					   LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT,
					   load, lookup);
		}
	}
	document_close(&document);
}

static bool cli_integer(const char *text, uint64_t most, uint64_t *value)
{
	if (text == NULL || *text == '\0')
		return false;
	uint64_t result = 0;
	for (; *text != '\0'; text++) {
		if (*text < '0' || *text > '9' || result > most / DECIMAL ||
		    (result == most / DECIMAL &&
		     (uint64_t)(*text - '0') > most % DECIMAL))
			return false;
		result = result * DECIMAL + (uint64_t)(*text - '0');
	}
	*value = result;
	return true;
}

/* A refused storage command names its reason and the usage on stderr. */
static int cli_refused(const char *reason)
{
	(void)fprintf(
		stderr,
		"storage: refused: %s\nusage: anemone storage-scan [--replica_external=0|1] | storage-device PATH "
		"[--replica_external=0|1] [--replica_floor_percent=N] | storage-room PATH BYTES [the same options] | "
		"storage-profile PATH | storage-report PATH [--metrics=1]\n",
		reason);
	return EXIT_FAILURE;
}
int storage_profile_cli(int count, char **arguments)
{
	if (count < CLI_FIRST)
		return cli_refused("no storage command");
	if (strcmp(arguments[1], "storage-scan") == 0) {
		const bool external = count == CLI_SECOND &&
				      strcmp(arguments[CLI_FIRST],
					     "--replica_external=1") == 0;
		if (count != CLI_FIRST && !external &&
		    !(count == CLI_SECOND &&
		      strcmp(arguments[CLI_FIRST], "--replica_external=0") ==
			      0))
			return cli_refused(
				"storage-scan takes only --replica_external=0|1");
		return local_device_scan(stdout, external) == STATUS_OK ?
			       EXIT_SUCCESS :
			       EXIT_FAILURE;
	}
	if (count < CLI_SECOND)
		return cli_refused("the command needs its path");
	const bool room = strcmp(arguments[1], "storage-room") == 0;
	if (room || strcmp(arguments[1], "storage-device") == 0) {
		uint64_t external = 0,
			 floor = LOCAL_DEVICE_DEFAULT_FLOOR_PERCENT, bytes = 0;
		int first_option = CLI_SECOND;
		if (room) {
			if (count < CLI_THIRD ||
			    !cli_integer(arguments[CLI_SECOND], UINT64_MAX,
					 &bytes))
				return cli_refused(
					"storage-room needs a byte count after its path");
			first_option++;
		}
		for (int index = first_option; index < count; index++) {
			const char *const argument = arguments[index];
			const size_t external_prefix =
				strlen("--replica_external=");
			const size_t floor_prefix =
				strlen("--replica_floor_percent=");
			if (strncmp(argument, "--replica_external=",
				    external_prefix) == 0) {
				if (!cli_integer(argument + external_prefix, 1,
						 &external))
					return cli_refused(
						"--replica_external takes 0 or 1");
			} else if (strncmp(argument, "--replica_floor_percent=",
					   floor_prefix) == 0) {
				if (!cli_integer(argument + floor_prefix,
						 PERCENT - 1u, &floor))
					return cli_refused(
						"--replica_floor_percent takes 0 to 99");
			} else {
				return cli_refused("an unknown option");
			}
		}
		struct local_device_description device = { 0 };
		const result_status status =
			local_device_describe(arguments[CLI_FIRST], &device);
		const char *reason = status != STATUS_OK ?
					     describe_refusal(&device) :
					     local_device_media_refusal(
						     &device, external != 0);
		char key[STORAGE_PROFILE_KEY_BYTES] = "";
		if (reason == NULL &&
		    storage_device_key_for(arguments[CLI_FIRST], external != 0,
					   key, &device) != STATUS_OK)
			reason = "hardware_identity_unavailable";
		if (reason == NULL && room)
			reason = local_device_space_refusal(
				device.total_bytes, device.available_bytes,
				bytes, (uint32_t)floor, 0, device.readonly);
		local_device_print(stderr, &device,
				   reason != NULL ? reason : "eligible", bytes,
				   (uint32_t)floor, 0, 0);
		char shown[REPLICA_PATH_CAPACITY];
		local_device_display(shown, sizeof(shown), device.mount);
		(void)printf(
			"{\"mount\":\"%s\",\"model\":\"%s\",\"transport\":\"%s\",\"filesystem\":\"%s\",",
			shown, device.model, device.transport,
			device.filesystem);
		(void)printf(
			"\"admitted\":%s,\"reason\":\"%s\",\"key\":\"%s\",\"physical\":\"%016" PRIx64
			"\",\"physical_devices\":[",
			reason == NULL ? "true" : "false",
			reason != NULL ? reason : "eligible", key,
			status == STATUS_OK ? local_device_group(&device) : 0);
		for (uint32_t index = 0;
		     status == STATUS_OK && index < device.count; index++)
			(void)printf("%s\"%016" PRIx64 "\"",
				     index == 0 ? "" : ",",
				     device.physical[index]);
		(void)printf("],\"total_bytes\":%" PRIu64
			     ",\"available_bytes\":%" PRIu64 "}\n",
			     status == STATUS_OK ? device.total_bytes : 0,
			     status == STATUS_OK ? device.available_bytes : 0);
		return EXIT_SUCCESS; /* Policy refusal is a decision, not a failed batch. */
	}
	const bool report = strcmp(arguments[1], "storage-report") == 0;
	const bool metrics_report =
		report && count == CLI_THIRD &&
		strcmp(arguments[CLI_SECOND], "--metrics=1") == 0;
	if ((!report && strcmp(arguments[1], "storage-profile") != 0) ||
	    (count != CLI_SECOND && !metrics_report))
		return cli_refused("an unknown storage command or option");
	struct storage_profile *const profile = calloc(1, sizeof(*profile));
	char reason[STORAGE_PROFILE_REASON_BYTES];
	if (profile == NULL)
		return cli_refused("no memory for the profile");
	if (report)
		saved_report(arguments[CLI_FIRST]);
	if (report)
		(void)fputs("storage: live profile revalidation\n", stderr);
	const bool valid =
		storage_profile_read(arguments[CLI_FIRST], profile, reason);
	if (report) {
		(void)fputs(
			"storage: live device scan (eligibility; explicit retune to change the saved set)\n",
			stderr);
		const result_status scanned =
			local_device_scan(NULL, profile->replica_external != 0);
		if (scanned != STATUS_OK)
			(void)fputs("storage: live scan unavailable\n", stderr);
	}
	if (!metrics_report)
		(void)printf(
			"{\"valid\":%s,\"reason\":\"%s\",\"replicas\":%" PRIu32
			",\"excluded\":%" PRIu32 "}\n",
			valid ? "true" : "false", reason, profile->count,
			profile->excluded);
	const result_status metrics_status = storage_profile_metrics(
		stdout, storage_profile_decisions(), metrics_report);
	free(profile);
	return valid && metrics_status == STATUS_OK ? EXIT_SUCCESS :
						      EXIT_FAILURE;
}

/* Explicit preparation shares the startup parser/hash, but never runs from startup. */
result_status prepare_hash(const void *bytes, size_t count,
			   char output[STORAGE_PROFILE_KEY_BYTES])
{
	if (bytes == NULL || output == NULL)
		return STATUS_INVALID_ARGUMENT;
	char value[STORAGE_PROFILE_KEY_BYTES];
	if (!hash_memory(bytes, count, value))
		return STATUS_INPUT_OUTPUT_ERROR;
	memcpy(output, value, sizeof(value));
	return STATUS_OK;
}

result_status prepare_hash_two(const void *first, const void *second,
			       size_t count,
			       char first_output[STORAGE_PROFILE_KEY_BYTES],
			       char second_output[STORAGE_PROFILE_KEY_BYTES])
{
	if (first == NULL || second == NULL || first_output == NULL ||
	    second_output == NULL)
		return STATUS_INVALID_ARGUMENT;
	unsigned char first_digest[SHA256_DIGEST_BYTES],
		second_digest[SHA256_DIGEST_BYTES];
	if (sha256_of_two(first, second, count, first_digest, second_digest) !=
	    STATUS_OK)
		return STATUS_INPUT_OUTPUT_ERROR;
	hash_text(first_digest, first_output);
	hash_text(second_digest, second_output);
	return STATUS_OK;
}

result_status prepare_manifest_read(const char *path,
				    struct prepare_manifest *output)
{
	if (path == NULL || output == NULL)
		return STATUS_INVALID_ARGUMENT;
	struct document document;
	if (!document_open(path, &document))
		return STATUS_INPUT_OUTPUT_ERROR;
	struct prepare_manifest *const value = calloc(1, sizeof(*value));
	uint32_t schema = 0;
	bool valid = value != NULL &&
		     field(&document, document.root, "schema", UINT32_MAX,
			   &schema) &&
		     schema == MANIFEST_SCHEMA &&
		     text_value(&document,
				member(&document, document.root, "source"),
				value->source, sizeof(value->source)) &&
		     text_value(&document,
				member(&document, document.root, "pack_id"),
				value->pack_identity,
				sizeof(value->pack_identity)) &&
		     number(member(&document, document.root, "bytes"),
			    INT64_MAX, &value->bytes) &&
		     value->bytes != 0 &&
		     number(member(&document, document.root, "chunk_bytes"),
			    MOST_CHUNK_BYTES, &value->chunk_bytes) &&
		     value->chunk_bytes >= PACK_BLOCK_BYTES &&
		     value->chunk_bytes % PACK_BLOCK_BYTES == 0 &&
		     value->bytes % PACK_BLOCK_BYTES == 0 &&
		     (value->bytes - 1u) / value->chunk_bytes <
			     PREPARE_CHUNKS_MOST;
	const struct json_value chunks =
		member(&document, document.root, "chunks");
	valid = valid && chunks.text != NULL &&
		chunks.text[chunks.begin] == '[';
	size_t cursor = chunks.begin;
	struct json_value hash;
	if (valid)
		value->chunks =
			(uint32_t)((value->bytes - 1u) / value->chunk_bytes +
				   1u);
	while (valid && json_next(chunks, &cursor, NULL, &hash)) {
		valid = value->known < value->chunks &&
			text_value(&document, hash, value->hashes[value->known],
				   STORAGE_PROFILE_KEY_BYTES) &&
			hex_key(value->hashes[value->known]);
		if (valid)
			value->known++;
	}
	document_close(&document);
	if (valid)
		*output = *value;
	free(value);
	return valid ? STATUS_OK : STATUS_CORRUPT_PACK;
}

/* The stages' sections (storage_profile.h). A schema 3 profile's top level holds the source and one member a stage;
 * each stage rewrites only its own member, and every other byte of the file stays as it was. stage_section_open opens
 * the profile at path and finds member_name's member and the source's real path; false, the document closed, with
 * reason missing, unreadable_or_invalid_json, not_a_stage_profile, no_<member>_section or <member>_section_malformed
 * (the source). The caller closes the document after a true. */
static bool stage_section_open(const char *path, const char *member_name,
			       struct document *profile,
			       struct json_value *section,
			       char source[REPLICA_PATH_CAPACITY],
			       char reason[STORAGE_PROFILE_REASON_BYTES])
{
	if (!document_open(path, profile))
		return reason_set(reason, access(path, F_OK) == 0 ?
						  "unreadable_or_invalid_json" :
						  "missing");
	uint32_t schema = 0;
	*section = member(profile, profile->root, member_name);
	const bool staged =
		field(profile, profile->root, "schema", UINT32_MAX, &schema) &&
		schema == STORAGE_PROFILE_STAGES_SCHEMA;
	const bool present = staged && section->text != NULL;
	if (present &&
	    text_value(profile, member(profile, profile->root, "source"),
		       source, REPLICA_PATH_CAPACITY) &&
	    source[0] == '/')
		return true;
	document_close(profile);
	if (!staged)
		return reason_set(reason, "not_a_stage_profile");
	(void)snprintf(reason, STORAGE_PROFILE_REASON_BYTES,
		       present ? "%s_section_malformed" : "no_%s_section",
		       member_name);
	return false;
}

/* The identity every stage's member holds: its schema (member_schema), enabled, the pack's header digest and size, and
 * the copies' device keys in order; false when one is missing or malformed. */
static bool section_identity(struct document *profile,
			     struct json_value section, uint32_t member_schema,
			     uint32_t *enabled,
			     char pack_identity[STORAGE_PROFILE_KEY_BYTES],
			     uint64_t *file_byte_count,
			     char (*keys)[STORAGE_PROFILE_KEY_BYTES],
			     uint32_t *copy_count)
{
	uint32_t schema = 0, count = 0;
	const struct json_value pack = member(profile, section, "pack");
	const struct json_value copies = member(profile, section, "copies");
	bool valid = field(profile, section, "schema", UINT32_MAX, &schema) &&
		     schema == member_schema &&
		     field(profile, section, "enabled", 1, enabled) &&
		     text_value(profile, member(profile, pack, "header_digest"),
				pack_identity, STORAGE_PROFILE_KEY_BYTES) &&
		     strlen(pack_identity) == STORAGE_PACK_IDENTITY_DIGITS &&
		     strspn(pack_identity, "0123456789abcdef") ==
			     STORAGE_PACK_IDENTITY_DIGITS &&
		     number(member(profile, pack, "file_byte_count"),
			    UINT64_MAX, file_byte_count) &&
		     copies.text != NULL && copies.text[copies.begin] == '[';
	size_t cursor = copies.begin;
	struct json_value copy;
	while (valid && json_next(copies, &cursor, NULL, &copy)) {
		valid = count < STORAGE_MAXIMUM_COPIES &&
			text_value(profile, member(profile, copy, "key"),
				   keys[count], STORAGE_PROFILE_KEY_BYTES) &&
			hex_key(keys[count]);
		count++;
	}
	*copy_count = count;
	return valid && count != 0;
}

/* Whether a section names this pack and exactly these copies, by their device keys in order; false with reason
 * <member>_section_stale_pack or <member>_section_stale_copies. A copy whose key is empty (none could be read) is
 * another copy. */
static bool section_names(const char *member_name, const char *pack_identity,
			  uint64_t file_byte_count,
			  const char (*keys)[STORAGE_PROFILE_KEY_BYTES],
			  uint32_t key_count, const struct pack_header *header,
			  const char (*copy_keys)[STORAGE_PROFILE_KEY_BYTES],
			  uint32_t copy_count,
			  char reason[STORAGE_PROFILE_REASON_BYTES])
{
	char identity[STORAGE_PROFILE_KEY_BYTES];
	(void)snprintf(identity, sizeof(identity), "%016" PRIx64,
		       header->header_digest);
	const bool same_pack = strcmp(identity, pack_identity) == 0 &&
			       header->file_byte_count == file_byte_count;
	bool same = same_pack && copy_count == key_count;
	for (uint32_t index = 0; same && index < copy_count; index++)
		same = strcmp(copy_keys[index], keys[index]) == 0;
	if (same)
		return true;
	(void)snprintf(reason, STORAGE_PROFILE_REASON_BYTES,
		       same_pack ? "%s_section_stale_copies" :
				   "%s_section_stale_pack",
		       member_name);
	return false;
}

bool storage_load_section_read(const char *path,
			       struct storage_load_section *output,
			       char reason[STORAGE_PROFILE_REASON_BYTES])
{
	if (path == NULL || output == NULL || reason == NULL)
		return false;
	struct document profile;
	struct json_value section;
	struct storage_load_section result = { 0 };
	if (!stage_section_open(path, STORAGE_LOAD_SECTION_NAME, &profile,
				&section, result.source, reason))
		return false;
	const struct json_value tp = member(&profile, section, "tp");
	uint64_t piece_bytes = 0;
	char split[SPLIT_NAME_BYTES] = "";
	const bool valid =
		section_identity(&profile, section, STORAGE_LOAD_SECTION_SCHEMA,
				 &result.enabled, result.pack_identity,
				 &result.file_byte_count, result.keys,
				 &result.copy_count) &&
		field(&profile, tp, "readers_per_copy", TP_LOAD_MAXIMUM_WORKERS,
		      &result.tp_readers_per_copy) &&
		number(member(&profile, tp, "piece_bytes"),
		       TP_HOST_LOAD_CHUNK_BYTES_MOST, &piece_bytes) &&
		field(&profile, tp, "host_threads", TP_LOAD_MAXIMUM_WORKERS,
		      &result.tp_host_threads) &&
		text_value(&profile, member(&profile, tp, "split"), split,
			   sizeof(split)) &&
		piece_bytes != 0 && piece_bytes % PACK_BLOCK_BYTES == 0;
	result.tp_split = TP_LOAD_SPLITS;
	for (uint32_t name = 0; name < TP_LOAD_SPLITS; name++)
		result.tp_split = strcmp(split, tp_load_split_names[name]) ==
						  0 ?
					  name :
					  result.tp_split;
	document_close(&profile);
	if (!valid || result.tp_split == TP_LOAD_SPLITS)
		return reason_set(reason, "load_section_malformed");
	result.tp_piece_bytes = piece_bytes;
	*output = result;
	return true;
}

bool storage_load_section_matches(const struct storage_load_section *section,
				  const struct pack_header *header,
				  const char *const *copy_paths,
				  uint32_t copy_count, bool external,
				  uint32_t host_threads, uint32_t split,
				  char reason[STORAGE_PROFILE_REASON_BYTES])
{
	/* each copy keyed by the drive of its bytes (an EXL3 map by its data drive), as the load stage keys it; a copy
	 * with no key (empty) matches none */
	char keys[STORAGE_MAXIMUM_COPIES][STORAGE_PROFILE_KEY_BYTES] = { { 0 } };
	for (uint32_t index = 0;
	     index < copy_count && index < STORAGE_MAXIMUM_COPIES; index++) {
		char bytes_path[EXL3_MAP_PATH_BYTES];
		struct local_device_description device;
		if (storage_device_key_for(
			    storage_device_path(copy_paths[index], bytes_path),
			    external, keys[index], &device) != STATUS_OK)
			keys[index][0] = '\0';
	}
	/* measured with other host threads or another split, its readers are no measure of this load's */
	return section_names(
		       STORAGE_LOAD_SECTION_NAME, section->pack_identity,
		       section->file_byte_count,
		       (const char (*)[STORAGE_PROFILE_KEY_BYTES])section->keys,
		       section->copy_count, header,
		       (const char (*)[STORAGE_PROFILE_KEY_BYTES])keys,
		       copy_count <= STORAGE_MAXIMUM_COPIES ?
			       copy_count :
			       STORAGE_MAXIMUM_COPIES + 1u,
		       reason) &&
	       ((section->tp_host_threads == host_threads &&
		 section->tp_split == split) ||
		reason_set(reason, "load_section_stale_shape"));
}

bool storage_engram_section_read(const char *path,
				 struct storage_engram_section *output,
				 char reason[STORAGE_PROFILE_REASON_BYTES])
{
	if (path == NULL || output == NULL || reason == NULL)
		return false;
	struct document profile;
	struct json_value section;
	struct storage_engram_section result = { 0 };
	if (!stage_section_open(path, STORAGE_ENGRAM_SECTION_NAME, &profile,
				&section, result.source, reason))
		return false;
	const struct json_value shape = member(&profile, section, "shape");
	const struct json_value step_io = member(&profile, section, "step_io");
	char interface[sizeof(STORAGE_ENGRAM_INTERFACE)] = "";
	const bool valid =
		section_identity(&profile, section,
				 STORAGE_ENGRAM_SECTION_SCHEMA, &result.enabled,
				 result.pack_identity, &result.file_byte_count,
				 result.keys, &result.copy_count) &&
		text_value(&profile, member(&profile, shape, "interface"),
			   interface, sizeof(interface)) &&
		strcmp(interface, STORAGE_ENGRAM_INTERFACE) == 0 &&
		field(&profile, shape, "steps", UINT32_MAX, &result.steps) &&
		field(&profile, shape, "slots", UINT32_MAX, &result.slots) &&
		field(&profile, step_io, "threads_per_copy",
		      STORAGE_FLIGHT_THREADS_PER_COPY_MOST,
		      &result.threads_per_copy) &&
		result.threads_per_copy != 0 &&
		field(&profile, step_io, "spin_us", ENGRAM_STEP_SPIN_US_MOST,
		      &result.spin_us) &&
		field(&profile, step_io, "pin", 1, &result.pin);
	document_close(&profile);
	if (!valid)
		return reason_set(reason, "engram_section_malformed");
	*output = result;
	return true;
}

bool storage_engram_section_matches(
	const struct storage_engram_section *section,
	const struct pack_header *header,
	const char (*copy_keys)[STORAGE_PROFILE_KEY_BYTES], uint32_t copy_count,
	uint32_t steps, uint32_t slots,
	char reason[STORAGE_PROFILE_REASON_BYTES])
{
	/* another flight shape is another measurement (serving's: 1 + STEP_PIPELINE_MOST steps of ENGRAM_STEP_SLOTS) */
	return section_names(
		       STORAGE_ENGRAM_SECTION_NAME, section->pack_identity,
		       section->file_byte_count,
		       (const char (*)[STORAGE_PROFILE_KEY_BYTES])section->keys,
		       section->copy_count, header, copy_keys, copy_count,
		       reason) &&
	       ((section->steps == steps && section->slots == slots) ||
		reason_set(reason, "engram_section_stale_shape"));
}

void storage_engram_choose(const struct storage_engram_section *section,
			   uint32_t ram_percent, bool explicit_threads,
			   bool explicit_spin, bool explicit_pin,
			   struct storage_engram_choice *choice)
{
	const bool applies = section != NULL &&
			     ram_percent < ENGRAM_RAM_PERCENT;
	choice->threads_from_profile = applies && !explicit_threads;
	choice->spin_from_profile = applies && !explicit_spin;
	choice->pin_from_profile = applies && !explicit_pin;
	if (choice->threads_from_profile)
		choice->threads_per_copy = section->threads_per_copy;
	if (choice->spin_from_profile)
		choice->spin_us = section->spin_us;
	if (choice->pin_from_profile)
		choice->pin = section->pin;
}

void storage_engram_section_apply(const struct storage_engram_serving *serving,
				  struct storage_engram_choice *choice,
				  char reason[STORAGE_PROFILE_REASON_BYTES])
{
	struct storage_engram_section section;
	const bool on_disk = serving->ram_percent < ENGRAM_RAM_PERCENT;
	(void)snprintf(reason, STORAGE_PROFILE_REASON_BYTES, "%s",
		       !on_disk ? "tables_in_ram" : "profile_off");
	bool usable = on_disk && serving->profile != NULL &&
		      serving->profile[0] != '\0' &&
		      storage_engram_section_read(serving->profile, &section,
						  reason) &&
		      storage_engram_section_matches(&section, serving->header,
						     serving->copy_keys,
						     serving->copy_count,
						     serving->steps,
						     serving->slots, reason);
	if (usable && section.enabled == 0) {
		(void)snprintf(reason, STORAGE_PROFILE_REASON_BYTES, "%s",
			       "engram_section_candidate");
		usable = false;
	}
	if (usable)
		reason[0] = '\0';
	storage_engram_choose(usable ? &section : NULL, serving->ram_percent,
			      serving->explicit_threads, serving->explicit_spin,
			      serving->explicit_pin, choice);
}

bool storage_load_section_reads_match(uint32_t copy_count, uint32_t map_reads,
				      char reason[STORAGE_PROFILE_REASON_BYTES])
{
	const uint32_t measured = copy_count == 1u ? EXL3_READS_DROPPED :
						     EXL3_READS_DIRECT_DROPPED;
	return map_reads >= EXL3_READ_MODES || map_reads == measured ||
	       reason_set(reason, "load_section_stale_shape");
}

void storage_load_choose(const struct storage_load_section *section,
			 bool explicit_readers, bool explicit_piece,
			 struct storage_load_choice *choice)
{
	choice->readers_from_profile = section != NULL && !explicit_readers;
	choice->piece_from_profile = section != NULL && !explicit_piece;
	if (choice->readers_from_profile)
		choice->tp_host_readers_per_copy = section->tp_readers_per_copy;
	if (choice->piece_from_profile)
		choice->tp_piece_bytes = section->tp_piece_bytes;
}

/* The profile's bytes for a stage write: STATUS_OK with *bytes NULL when there is no file. A file that is no schema 3
 * profile of this source is refused by name, so a stage never replaces another tuning's profile. */
MUST_CHECK static result_status stage_profile_bytes(const char *path,
						    const char *real_source,
						    struct document *document,
						    const char **refusal)
{
	struct stat file;
	if (lstat(path, &file) != 0 && errno == ENOENT)
		return STATUS_OK;
	char source[REPLICA_PATH_CAPACITY];
	uint32_t schema = 0;
	if (!document_open(path, document)) {
		*refusal =
			"the profile is unreadable, not this user's alone or no JSON object";
		return STATUS_INVALID_ARGUMENT;
	}
	*refusal =
		!field(document, document->root, "schema", UINT32_MAX,
		       &schema) ||
				schema != STORAGE_PROFILE_STAGES_SCHEMA ?
			"the profile is no stage profile (schema 3): give the stage another --profile" :
		!text_value(document,
			    member(document, document->root, "source"), source,
			    sizeof(source)) ||
				strcmp(source, real_source) != 0 ?
			"the profile tunes another source: give the stage another --profile" :
			NULL;
	if (*refusal == NULL)
		return STATUS_OK;
	/* a refused profile's document is closed here: its callers return at once */
	document_close(document);
	return STATUS_INVALID_ARGUMENT;
}

/* Writes text[0, begin), then insert, then text[end, length) into path, atomically. */
MUST_CHECK static result_status
spliced_write(const char *path, const char *text, size_t length, size_t begin,
	      size_t end, const char *insert, size_t insert_length)
{
	struct prepare_file file;
	result_status status = prepare_file_open(path, &file);
	if (status != STATUS_OK)
		return status;
	if (fwrite(text, 1, begin, file.stream) != begin ||
	    fwrite(insert, 1, insert_length, file.stream) != insert_length ||
	    fwrite(text + end, 1, length - end, file.stream) != length - end) {
		prepare_file_abort(&file);
		return STATUS_INPUT_OUTPUT_ERROR;
	}
	return prepare_file_commit(&file);
}

/* A stage's member name: one of the stages' (storage_profile.h), so the JSON key needs no escape. */
static bool stage_member_known(const char *member_name)
{
	return member_name != NULL &&
	       (strcmp(member_name, STORAGE_LOAD_SECTION_NAME) == 0 ||
		strcmp(member_name, STORAGE_ENGRAM_SECTION_NAME) == 0);
}

result_status storage_stage_section_write(const char *path,
					  const char *real_source,
					  const char *member_name,
					  const char *section, size_t length,
					  const char **refusal)
{
	if (path == NULL || real_source == NULL || section == NULL ||
	    refusal == NULL || !prepare_utf8_text(real_source) ||
	    !stage_member_known(member_name))
		return STATUS_INVALID_ARGUMENT;
	*refusal = NULL;
	struct document document = { 0 };
	result_status status =
		stage_profile_bytes(path, real_source, &document, refusal);
	if (status != STATUS_OK)
		return status;
	if (document.bytes == NULL) {
		char *created = NULL;
		size_t created_length = 0;
		FILE *const stream = open_memstream(&created, &created_length);
		if (stream == NULL)
			return STATUS_OUT_OF_MEMORY;
		(void)fprintf(stream, "{\"schema\":%u,\"source\":",
			      STORAGE_PROFILE_STAGES_SCHEMA);
		prepare_json_text(stream, real_source);
		(void)fprintf(stream, ",\"%s\":", member_name);
		(void)fwrite(section, 1, length, stream);
		(void)fputs("}\n", stream);
		status = fclose(stream) == 0 ?
				 spliced_write(path, created, created_length,
					       created_length, created_length,
					       "", 0) :
				 STATUS_OUT_OF_MEMORY;
		free(created);
		return status;
	}
	const struct json_value root = document.root;
	const struct json_value kept = member(&document, root, member_name);
	char *inserted = NULL;
	size_t inserted_length = 0;
	FILE *const stream = open_memstream(&inserted, &inserted_length);
	if (stream == NULL) {
		document_close(&document);
		return STATUS_OUT_OF_MEMORY;
	}
	if (kept.text == NULL)
		(void)fprintf(stream, ",\"%s\":", member_name);
	(void)fwrite(section, 1, length, stream);
	/* an existing member's value is replaced; else the member goes in before the root's closing brace */
	const size_t begin = kept.text != NULL ? kept.begin : root.end - 1u;
	const size_t end = kept.text != NULL ? kept.end : root.end - 1u;
	status = fclose(stream) == 0 ?
			 spliced_write(path, document.bytes,
				       strlen(document.bytes), begin, end,
				       inserted, inserted_length) :
			 STATUS_OUT_OF_MEMORY;
	free(inserted);
	document_close(&document);
	return status;
}

result_status storage_load_section_write(const char *path,
					 const char *real_source,
					 const char *section, size_t length,
					 const char **refusal)
{
	return storage_stage_section_write(path, real_source,
					   STORAGE_LOAD_SECTION_NAME, section,
					   length, refusal);
}

result_status storage_stage_section_writable(const char *path,
					     const char *real_source,
					     const char **refusal)
{
	if (path == NULL || real_source == NULL || refusal == NULL)
		return STATUS_INVALID_ARGUMENT;
	*refusal = NULL;
	struct document document = { 0 };
	const result_status status =
		stage_profile_bytes(path, real_source, &document, refusal);
	if (status == STATUS_OK && document.bytes != NULL)
		document_close(&document);
	return status;
}

result_status storage_load_section_writable(const char *path,
					    const char *real_source,
					    const char **refusal)
{
	return storage_stage_section_writable(path, real_source, refusal);
}

result_status storage_stage_section_qualify(const char *path,
					    const char *real_source,
					    const char *member_name,
					    const char **refusal)
{
	if (path == NULL || real_source == NULL || refusal == NULL ||
	    !stage_member_known(member_name))
		return STATUS_INVALID_ARGUMENT;
	*refusal = NULL;
	struct document document = { 0 };
	result_status status =
		stage_profile_bytes(path, real_source, &document, refusal);
	if (status != STATUS_OK)
		return status;
	if (document.bytes == NULL) {
		*refusal =
			"there is no profile to qualify: run the stage first";
		return STATUS_INVALID_ARGUMENT;
	}
	const struct json_value section =
		member(&document, document.root, member_name);
	const struct json_value enabled = member(&document, section, "enabled");
	const struct json_value qualification =
		member(&document, section, "qualification");
	static const char qualified[] = "1,\"qualification\":\"qualified\"",
			  between[] = ",\"qualification\":";
	/* the two members as the stage writes them, side by side: enabled 0, then the candidate's qualification */
	document.scratch.used = 0;
	const bool candidate =
		enabled.text != NULL && qualification.text != NULL &&
		enabled.end - enabled.begin == 1u &&
		document.bytes[enabled.begin] == '0' &&
		qualification.begin - enabled.end == sizeof(between) - 1u &&
		memcmp(document.bytes + enabled.end, between,
		       sizeof(between) - 1u) == 0 &&
		json_string_is(&document.scratch, qualification, "candidate");
	if (!candidate) {
		*refusal =
			strcmp(member_name, STORAGE_ENGRAM_SECTION_NAME) == 0 ?
				"the profile holds no candidate engram section" :
				"the profile holds no candidate load section";
		status = STATUS_INVALID_ARGUMENT;
	} else
		status = spliced_write(path, document.bytes,
				       strlen(document.bytes), enabled.begin,
				       qualification.end, qualified,
				       sizeof(qualified) - 1u);
	document_close(&document);
	return status;
}

result_status storage_load_section_qualify(const char *path,
					   const char *real_source,
					   const char **refusal)
{
	return storage_stage_section_qualify(
		path, real_source, STORAGE_LOAD_SECTION_NAME, refusal);
}
