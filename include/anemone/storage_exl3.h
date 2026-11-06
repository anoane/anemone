/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Read-only EXL3 virtual pack. A manifest maps sorted disjoint virtual ranges to
 * regular source files; gaps read as zero. Selected once at load by .exl3map suffix.
 * Source paths are bounded to 511 bytes, source count to 128, segments to 262144.
 * No source is writable and no manifest pointer is borrowed. Copy 0 is the map; copies 1.. (turbo load's
 * replicas, replicas.h) are flat files of its logical bytes, read through storage_direct (O_DIRECT).
 * Open validates all source and virtual bounds before publishing the reader.
 * Refusals leave outputs unchanged; close accepts NULL. Concurrent reads use pread.
 * The map must be produced from the pinned checkpoint; extent digests still gate load.
 * NVMe source read bounds share the native reader's admission policy; unknown
 * bounds retain the seam's accepted warning policy. Nonblocking open refuses FIFOs. */
#ifndef STORAGE_EXL3_H
#define STORAGE_EXL3_H

#include "anemone/storage_seam.h"

#define EXL3_MAP_PATH_BYTES 512u
#define EXL3_MAP_MAXIMUM_SOURCES 128u
#define EXL3_MAP_MAXIMUM_SEGMENTS 262144u
#define EXL3_MAP_HEADER_BYTES 64u
#define EXL3_MAP_SEGMENT_BYTES 32u
#define EXL3_MAP_BLOCK_BYTES 4096u

extern const struct storage_operations_record storage_exl3;
/* How a map reads its sources; set before reader_open. BUFFERED: through the page cache, as before. DROPPED: the same
 * reads, then each piece's whole pages leave the page cache (POSIX_FADV_DONTNEED): hits stay hits, but a load does not
 * keep a second copy of what it holds (the host arena) or has sent to its cards, so the arena's memory needs no
 * reclaim (3 Nov: a serial load's arena filled at 2.2 GB/s while memory was free, then ~0.17 GB/s with 4.8 GB free and
 * 24% of it on small pages). DIRECT: each source opened once more with O_DIRECT, and a piece that meets its direct
 * alignment (offset, size, address) read that way (the native pack's extents: 228.5 GB of the EXL3 TP4 map); the
 * rest buffered. Without a call: DIRECT (anemone replicate leaves the cache as it was); the engine's --map_reads sets
 * it (default DIRECT_DROPPED with the TP load's copy split: the fastest TP4
 * load, 4 Nov). */
enum exl3_read_mode {
	EXL3_READS_BUFFERED,
	EXL3_READS_DROPPED,
	EXL3_READS_DIRECT,
	EXL3_READS_DIRECT_DROPPED, /* DIRECT where aligned, DROPPED for the rest: a load from cold drives */
	EXL3_READ_MODES
};
void storage_exl3_read_mode(enum exl3_read_mode mode);
const struct storage_operations_record *storage_for_path(const char *path);

#endif
