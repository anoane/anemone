/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Isolated private-helper regression (r18 review): a copy journal and a verification receipt name the device by its
 * hardware key, never by its device number, which a reboot may renumber (NVMe partitions take minors in probe
 * order); every other field they bind still changes them. */
#include "../../../source/replica_prepare.c"

#define TEST_CHUNKS 2u
#define TEST_INODE 4242u
#define TEST_FIRST_DEVICE 0x10301u /* 259:1 */
#define TEST_RENUMBERED_DEVICE \
	0x10305u /* the same partition as 259:5 after a reboot */
#define TEST_MTIME_SECONDS 1762560000
#define TEST_CTIME_SECONDS 1762560001
#define TEST_PACK "0000000000000001"

static const char test_key[] =
	"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
static const char other_key[] =
	"1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

/* Whether the receipt of file differs from reference. */
static bool receipt_differs(const struct stat *file,
			    const struct prepare_manifest *manifest,
			    const char *key, const char *reference)
{
	char text[RECEIPT_BYTES];
	return receipt_text(file, manifest, key, text, sizeof(text)) &&
	       strcmp(text, reference) != 0;
}

/* Whether the journal expected for file and key differs from reference in a field it compares. */
static bool journal_differs(const struct stat *file,
			    const struct prepare_manifest *manifest,
			    const char *key, const struct copy_state *reference)
{
	struct copy_state state;
	copy_state_expected(&state, manifest, key, file);
	return memcmp(&state, reference, offsetof(struct copy_state, bitmap)) !=
	       0;
}

int main(void)
{
	struct prepare_manifest *const manifest = calloc(1, sizeof(*manifest));
	if (manifest == NULL)
		return 1;
	manifest->chunk_bytes = PREPARE_CHUNK_BYTES;
	manifest->chunks = manifest->known = TEST_CHUNKS;
	manifest->bytes = (uint64_t)TEST_CHUNKS * PREPARE_CHUNK_BYTES;
	for (uint32_t chunk = 0; chunk < TEST_CHUNKS; chunk++)
		(void)snprintf(manifest->hashes[chunk],
			       sizeof(manifest->hashes[chunk]), "%064x",
			       chunk + 1u);
	(void)snprintf(manifest->pack_identity, sizeof(manifest->pack_identity),
		       "%s", TEST_PACK);
	struct stat first = { 0 };
	first.st_dev = TEST_FIRST_DEVICE;
	first.st_ino = TEST_INODE;
	first.st_size = (off_t)manifest->bytes;
	first.st_mtim.tv_sec = TEST_MTIME_SECONDS;
	first.st_ctim.tv_sec = TEST_CTIME_SECONDS;
	struct stat renumbered = first;
	renumbered.st_dev = TEST_RENUMBERED_DEVICE;
	struct copy_state journal, after_reboot;
	copy_state_expected(&journal, manifest, test_key, &first);
	copy_state_expected(&after_reboot, manifest, test_key, &renumbered);
	char before[RECEIPT_BYTES], after[RECEIPT_BYTES];
	bool passed = memcmp(&journal, &after_reboot, sizeof(journal)) == 0 &&
		      receipt_text(&first, manifest, test_key, before,
				   sizeof(before)) &&
		      receipt_text(&renumbered, manifest, test_key, after,
				   sizeof(after)) &&
		      strcmp(before, after) == 0;
	/* the journal: the hardware key and the inode */
	struct stat other = first;
	other.st_ino++;
	passed = passed &&
		 journal_differs(&first, manifest, other_key, &journal) &&
		 journal_differs(&other, manifest, test_key, &journal);
	/* the receipt: the hardware key, inode, size, mtime, ctime, a manifest hash and the pack */
	passed = passed &&
		 receipt_differs(&first, manifest, other_key, before) &&
		 receipt_differs(&other, manifest, test_key, before);
	other = first;
	other.st_size--;
	passed = passed && receipt_differs(&other, manifest, test_key, before);
	other = first;
	other.st_mtim.tv_nsec++;
	passed = passed && receipt_differs(&other, manifest, test_key, before);
	other = first;
	other.st_ctim.tv_nsec++;
	passed = passed && receipt_differs(&other, manifest, test_key, before);
	manifest->hashes[1][0] = 'f';
	passed = passed && receipt_differs(&first, manifest, test_key, before);
	manifest->hashes[1][0] = '0';
	manifest->pack_identity[0] = '1';
	passed = passed && receipt_differs(&first, manifest, test_key, before);
	free(manifest);
	return passed ? 0 : 1;
}
