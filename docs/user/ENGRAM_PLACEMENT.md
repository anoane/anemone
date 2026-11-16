<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Experimental Engram placement

All immutable Engram tables remain in RAM by default when they fit. Without `--engram_ram_percent` (the default
`auto`), generate and serve compare the plan's host memory with MemAvailable before the load. They keep every table
in RAM when it fits. Otherwise they take the largest whole percent that fits, with the rest read from NVMe, and else
0. The run is refused before the load when the model's own host memory does not fit, or when the tables need disk
copies and none is admitted. Two stderr lines give the percent with its source (the knob, the storage profile or
auto) and the whole budget: tables, disk store, other host-only units, pinned host experts, load staging and the
headroom, max(8 GiB, MemTotal / 32). PP's load staging is 8 slots of the load's piece (`--load_piece_mib`, 16 MiB by
default) a card; the cards are counted after the fit, so without `--card_order` it charges 16 cards, an upper bound
the line names. Below 100 % the load's host arena still spans every table, the disk tables' never touched, so it
is mapped without commit accounting (MAP_NORESERVE): Linux refuses one accounted mapping larger than RAM + swap,
which is the host the disk tables are for. At 100 it is mapped as before. `--engram_fit_only=1` (generate) prints them and stops before any card is
touched. Outputs are the same at every percent, so this is a capacity decision; no profile axis selects it. The
following raw options are experimental; no four-axis selection enables disk placement. Native numerical, state and
paired serving qualification is pending.

| Option | Meaning |
| --- | --- |
| `--engram_ram_percent=auto` | The default: the fit above decides 100, a share or 0. |
| `--engram_ram_percent=100` | Existing all-RAM loader and gathers. |
| `--engram_ram_percent=50` | Keep the first half of each table's padded blocks in owned RAM, rounding down to a block. |
| `--engram_ram_percent=0` | Fetch all table blocks from disk, with optional caching. |
| `--engram_cache_mib=0..32768` | Additional shared block-cache storage, rounded down to sixteen 4096-byte blocks. Default 0. |
| `--engram_io=0` | Retained serial O_DIRECT reader. |
| `--engram_io=1` | Eight prepared io_uring lanes, registered files and batches of up to 32 independent 4096-byte reads. Refuses unsupported setup or unknown direct alignment. |
| `--engram_sources=PACK[,PACK...]` | Optional independent flat table sources. Source model weights are ignored. |
| `--turbo-nvme-engrams=1` | Interleave runtime reads among table sources on distinct physical devices. Default 0. Requires disk placement and at least two sources. |
| `--engram_step_io=1` | Read a step's blocks at once when its row ids are known, on I/O threads per copy that verify each block and fall back to another copy. The default when serving PP5 or TP4_PRO from disk tables; `--engram_step_io=0` keeps the per-row gathers, the default elsewhere. The startup line `engram step io:` names the path. |
| `--engram_io_threads=1..4`, `--engram_spin_us=0..100000`, `--engram_io_pin=0\|1` | Step I/O threads per copy (default 2), their busy-poll window after the last read before sleeping (default 200 µs), and pinning to each drive's completion CPU and its sibling (default 1). |
| `--engram_hot_set=FILE --engram_hot_sha256=HEX [--engram_hot_mib=N]` | A static hot set of table blocks held in RAM and served like the RAM prefix (default none, at most 512 MiB). |

Serving PP5 (request 7, every stream identical to all-RAM): with step I/O, a 131k-token cold prefill took 27.4 s
against 27.2 s with the tables in RAM and 47.9 s with per-row gathers; decode at one session ran 2.2 % slower than
all-RAM and at four and eight sessions equal, on about 0.8 cores of I/O.

Step I/O gives each block read its own deadline: the cards' 10 s wait limit over twice (copies + 1), 1.25 s with
three copies, so a stalled read is read again on another copy before a pipeline that sees no progress fails the step.
The startup line shows it (`read limit 1250 ms`), and also how many rings read without registered buffers when the
locked-memory limit refused them. A copy whose read passed its deadline takes no more reads; the slots the kernel may
still write into stay out until their late completions land, and then come back. After a backoff (1 s, doubled after
each failed attempt up to about 8.5 minutes), or at once when the copy answers a verified read, its I/O threads set up
fresh rings, and one probe read on an I/O thread, in a slot of that copy's own, asks the drive before any step reads
the copy again; no serving thread waits on the probe, and a probe that never returns holds back only its own copy. A
copy that stalls again within a minute of coming back keeps its doubled backoff, so a drive that keeps stalling is
asked ever less often. A step record whose read never returned keeps its staging at shutdown, which is reported.

These settings affect the large immutable model tables only. Compressed token
maps, hashing constants, prompt lookup drafter/history, DSpark weights,
attention state and other model weights retain their existing ownership.

The disk route reads flat table sources on admitted local NVMe or ATA SSD
filesystems. Without `--engram_sources`, it reuses the model loader's flat pack
copies. An EXL3 model map requires explicit flat table sources. Each independent
source may have another model-weight layout: startup compares every compressed
token-map/hash constant, row count, table byte count and extent digest, then
translates the logical table offsets to that source's actual extents. Source
weights are never supplied to the model loader. This permits reuse of identical
Engram tables in original and EXL3 packs. No files are copied or created while
serving. Source paths use commas as separators and cannot themselves contain
commas; duplicate physical devices are refused.

Device selection rejects unknown topology, removable/rotational media, SD,
USB, Thunderbolt/USB4 enclosures (without the external override) and network
filesystems. Composite dm/md devices resolve to their physical members, which
must each qualify; partitions and bind mounts of one physical device share an
identity. No speed probe is run: a slow device is not refused by selection,
and measured shares come only from autotune.
New replica writes require at least 20% of total filesystem space to remain
available, in addition to the existing explicit free-space floor. Existing
read-only copies can be used even when their filesystem is already below that
floor. Original all-RAM arithmetic is unchanged by these selection checks.

A row contains 256 value bytes and eight scale bytes. Fifteen rows occupy the
start of each 4096-byte block. Preparation reads each table with 16 MiB of
workspace, validates the original extent digest, fills the chosen RAM prefix,
and records an eight-byte digest per stored block. Disk reads must match the
recorded block digest. I/O errors, short reads and changed blocks retry once on
the same copy and then try another. A failed batched read has already marked its
copy, so the other copies are tried first and that copy last. A copy whose read
failed is tried last but never retired, and a verified read clears that mark.
When every copy fails a block, that gather returns failure with one error line
on stderr, in every build, naming the table, the block and the status, and
never uses partial output.
The cache uses fixed direct-mapped slots with sixteen locks; cache metadata and
block digests are additional to its byte limit. Eight io_uring buffer lanes use
1 MiB total; a gather uses at most 128 KiB of stack, and at most 64 requested
rows are accepted. No token-path allocation occurs. Runtime read completion and
ring-lock waits have explicit limits. A ring timeout quarantines its owned
buffers/mappings until process exit, so unfinished I/O cannot reference a
returned caller buffer. Store destruction requires all gather callers to retire.

The existing loader still reserves its original host virtual address range;
externally owned table ranges remain untouched and uncommitted. Resident RAM
consists of the selected prefixes, block digests, bounded cache and other normal
engine allocations. Address-space limits can therefore still prevent startup
with disk placement. A stricter virtual-address compaction is not implemented.

Measure process-start-to-serving separately from runtime gather cost: disk
preparation includes a complete table verification scan. Keep one-time replica
copying outside ordinary startup comparisons. Cold buffered tests require
owned files and file-local eviction; never use system-wide drop_caches. Use
O_DIRECT on borrowed packs when avoiding page cache. Disk placement primarily
trades RAM capacity for I/O; a component I/O improvement does not establish a
TTFT, decode, prefill or concurrent serving improvement.
