/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * replicas.h: turbo load's replicas of a pack. A replica is a flat copy of a pack's bytes (an EXL3 map's logical
 * bytes) on another drive; a load adds it as a copy and spreads its reads over the copies (storage_seam.h: the
 * caller picks the copy of each read), and every extent it reads is still checked against the pack's digest.
 *
 * `anemone replicate [--rate_mib=N] [--floor_gib=N] [--prune=0|1] SOURCE [DIRECTORY ...]` writes one replica into each DIRECTORY,
 * or with none into <mount>/anemone/replicas of every other local drive (ext4, xfs, btrfs or f2fs on a /dev device,
 * not under /boot) where it fits: the filesystem stays at most REPLICA_MOST_USED_PERCENT full and keeps
 * --floor_gib (default REPLICA_DEFAULT_FLOOR_GIB) free. It reads the source at most --rate_mib MiB/s
 * (default REPLICA_DEFAULT_RATE_MIB; 0: no limit), writes <header digest>.anemone.partial with direct
 * writes, reads it back with direct reads and checks its header and tables (pack_open_entry), that its header is
 * the source's, and every extent's digest; only then is it renamed and its receipt written (<replica>.receipt: the
 * source's real path, size and header digest). A replica that failed or was interrupted keeps its .partial name and
 * is never used. --prune=1 first removes, in each directory, the receipted replicas of the same source whose header
 * digest is another (the pack has changed since). Run it outside the GPU queue: it competes with loads for the
 * source drive.
 *
 * replicas_find writes the paths of the replicas whose receipts name the source (its real path), its size
 * and its header digest, at most capacity of them, and returns their number. It reads only /proc/self/mounts and the
 * receipts; it opens no replica.
 */
#ifndef REPLICAS_H
#define REPLICAS_H

#include <limits.h>
#include <stdint.h>

#include "anemone/pack.h"
#include "anemone/storage_seam.h"

#define REPLICAS_MOST (STORAGE_MAXIMUM_COPIES - 1)
#define REPLICA_PATH_CAPACITY PATH_MAX
#define REPLICA_MOST_USED_PERCENT 90u
#define REPLICA_DEFAULT_FLOOR_GIB 100u
#define REPLICA_DEFAULT_RATE_MIB 2048u

uint32_t replicas_find(const char *source_path,
		       const struct pack_header *header,
		       char (*paths)[REPLICA_PATH_CAPACITY], uint32_t capacity);
int replicate_cli(int argument_count, char **arguments);

#endif
