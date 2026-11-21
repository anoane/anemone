/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * replicas.h: turbo load's replicas of a pack. A replica is a flat copy of a pack's bytes (an EXL3 map's logical
 * bytes) on another drive; a load adds it as a copy and spreads its reads over the copies (storage_seam.h: the
 * caller picks the copy of each read), and every extent it reads is still checked against the pack's digest.
 *
 * `anemone replicate [--rate_mib=N] [--floor_gib=N] [--prune=0|1] SOURCE [DIRECTORY ...]` writes one replica into
 * each DIRECTORY, or with none into <mount>/anemone_engram_rnd/replicas of every other filesystem (ext4, xfs or f2fs
 * on a /dev device; btrfs is passed over with one line, not under /boot) where it fits: the filesystem stays at most
 * REPLICA_MOST_USED_PERCENT full and keeps
 * --floor_gib (default REPLICA_DEFAULT_FLOOR_GIB) free. It reads the source at most --rate_mib MiB/s
 * (default REPLICA_DEFAULT_RATE_MIB; 0: no limit), writes <header digest>.anemone.replicate.partial
 * (REPLICA_PARTIAL_SUFFIX) with direct writes, reads it back with direct reads and checks its header and tables
 * (pack_open_entry), that its header is
 * the source's, and every extent's digest; only then is it renamed and its receipt written (<replica>.receipt: the
 * source's real path, size and header digest). A refused or failed copy removes its owned partial and is never
 * used. External devices require --replica_external=1; network/virtual devices are always excluded.
 * --replica_floor_percent defaults to 20. --prune=1 first removes, in each directory, the receipted replicas of the
 * same source whose header digest is another (the pack has changed since). Run it in an admitted queue copy window:
 * it competes with loads for the source drive.
 *
 * `anemone replicate --adopt=COPY [--replica_external=0|1] SOURCE` adopts an existing full copy of SOURCE (one made
 * before receipts, or by hand) on another drive than SOURCE's, without copying a byte: replicas_adopt below.
 *
 * replicas_find writes the paths of the replicas whose receipts name the source (its real path), its size
 * and its header digest, at most capacity of them, and returns their number. It reads /proc/self/mounts, stats the
 * points of the ext4, xfs and f2fs filesystems on /dev devices (a btrfs subvolume has no block-device topology),
 * describes them from sysfs, and reads receipts only on the mounts the media policy admits (internal and not
 * rotational; external with --replica_external=1). On each it reads, in order, $HOME/anemone_engram_rnd/replicas when
 * HOME is on that filesystem (where autotune places the root filesystem's replicas: a mount point owned by root takes
 * none), then <mount>/anemone_engram_rnd/replicas and <mount>/anemone/replicas. It opens no replica, but admits one
 * only when the file the receipt stands for is a regular file (no link) of exactly the source's size; a receipt beside
 * a missing, linked or short file is printed excluded (replica_missing_short_or_unverified). The receipt names the
 * source and the pack, not the replica's inode or times, so a same-size replacement is caught only by the extent
 * digests every read checks.
 *
 * replicas_find_policy and replicas_find_mounts search beside the copy_count copies a load was given (copies[0] the
 * source, whose receipts they match; serve's turbo load beside explicit copies, 17 Nov 2026): a replica on a drive that
 * holds any given copy is refused as duplicate_physical_device, as one on the source's drive is. A given copy no
 * device topology describes (an anonymous st_dev: btrfs, ZFS) or whose medium names no drive (overlay, a loop or
 * virtual block device, FUSE, a network filesystem), the source too, ends the search at its first receipted replica
 * with one stderr line naming it (unknown_physical_topology): no replica is admitted beside it, as its drive cannot be
 * compared. A copy on a RAM-backed filesystem (tmpfs, ramfs, by name) shares no drive, so the search runs beside it.
 * replicas_find is the search beside the source alone.
 *
 * replicas_reader_open opens a reader over the given_count given copies and the count replicas together, one open when
 * all open. When that open is refused it keeps, greedily in their order, each replica that opens together with the
 * copies already kept: a replica whose open fails beside them is left out, with one line on output naming it and the
 * open's status, and the next is tried. Per-copy resources (registered buffers, locked memory, queue depth) can fail
 * only in combination, so the kept set is one that did open, in at most 1 + count opens.
 * copies_out (given_count + count entries) receives the given copies and the kept replicas; reader_out the reader over
 * them, or NULL when none was kept (the caller keeps its reader over the given copies). Returns the number kept.
 */
#ifndef REPLICAS_H
#define REPLICAS_H

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "engine/formats/pack/pack.h"
#include "engine/storage/storage_seam.h"

#define REPLICAS_MOST (STORAGE_MAXIMUM_COPIES - 1)
#define REPLICA_PATH_CAPACITY PATH_MAX
/* anemone replicate's in-progress copy beside its target: never autotune's resumable checkpoint name, which it
 * would otherwise remove before its own copy; still a .partial, so no profile or scan admits it */
#define REPLICA_PARTIAL_SUFFIX ".replicate.partial"
#define REPLICA_MOST_USED_PERCENT 80u
#define REPLICA_DEFAULT_FLOOR_GIB 100u
#define REPLICA_DEFAULT_RATE_MIB 2048u
/* A replica's receipt beside it (<replica>.receipt), which the search matches word for word: the source's real path,
 * its size and its header digest. Written only once every byte of the replica was read and checked: by anemone
 * replicate after its copy, by --adopt, and by autotune's copy stage after a full verification. */
#define REPLICA_RECEIPT_SUFFIX ".receipt"
MUST_CHECK bool replicas_receipt_text(char *output, size_t capacity,
				      const char *real_source,
				      uint64_t byte_count,
				      uint64_t header_digest);

uint32_t replicas_find(const char *source_path,
		       const struct pack_header *header,
		       char (*paths)[REPLICA_PATH_CAPACITY], uint32_t capacity);
uint32_t replicas_find_policy(const char *const *copies, uint32_t copy_count,
			      const struct pack_header *header,
			      char (*paths)[REPLICA_PATH_CAPACITY],
			      uint32_t capacity, bool external);
/* replicas_find_policy over the mount table stream mounts (the product's is /proc/self/mounts). */
uint32_t replicas_find_mounts(FILE *mounts, const char *const *copies,
			      uint32_t copy_count,
			      const struct pack_header *header,
			      char (*paths)[REPLICA_PATH_CAPACITY],
			      uint32_t capacity, bool external);
uint32_t
replicas_reader_open(const struct storage_operations_record *operations,
		     const char *const *given, uint32_t given_count,
		     const char (*replicas)[REPLICA_PATH_CAPACITY],
		     uint32_t count, const char **copies_out,
		     struct storage_reader **reader_out, FILE *output);
/* Adopts path, an existing file that should be a full copy of the source (real_source, header) on the filesystem
 * mounted at point: it must be a regular file the user owns (not a symbolic link), of the source's size, and not
 * the source. Its search directory is the first of the search's directories on that filesystem (above) that exists,
 * else the first that can be made. Then every block is read with direct reads and checked: pack_open_entry checks
 * the header and tables, the header must be the source's, and every extent must have its digest; the file must not
 * change while it is read. Only then is it hard-linked into the directory under the replica's name (path keeps its
 * own name; no byte is copied), and its receipt written (partial, synced, renamed, the directory synced). A file
 * already adopted (the name is its inode and the receipt holds) is not read again. A copy that fails the check gets
 * no name and no receipt, and the first differing extent is named on stderr and in differing_out (UINT32_MAX when
 * none differed: a header, table or read failure). The caller applies the media and distinct-drive policy. */
enum replica_adoption {
	REPLICA_ADOPTED,
	REPLICA_ADOPT_REFUSED, /* the file or the directory policy: nothing read */
	REPLICA_ADOPT_FAILED /* read and refused, or a link or receipt that could not be written */
};
enum replica_adoption replicas_adopt(const char *path, const char *point,
				     const char *real_source,
				     const struct pack_header *header,
				     uint32_t *differing_out);
int replicate_cli(int argument_count, char **arguments);

#endif
