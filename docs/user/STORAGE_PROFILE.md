<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Persistent storage tuning and replica policy

The owner recommends running `anemone autotune` once before serving, and again
after hardware changes. The engine discovers devices, copies missing replicas,
tunes the two actions and atomically saves a profile for later automatic reload.
Serving never invokes this command or starts a benchmark itself.

```sh
anemone autotune /srv/model.anemone --tokens=/srv/prompt_ids.u32 --prompt=1024 --profile=/srv/runtime/storage-profile.json --seconds=1650
```

Loading the model and looking up n-gram rows are separate operations, so autotune is being split into separate
stages: replica preparation (copy), the load stage and the Engram stage. Each stage has its own parameters,
mechanism, state and profile section. `--stage=copy` runs the copy stage alone:

```sh
anemone autotune /srv/model.anemone --stage=copy --target=/srv/anemone_engram_rnd/replicas/<identity>.anemone --profile=/srv/runtime/storage-profile.json
```

It prepares and verifies the replicas exactly as below, writes their verification and replica-search receipts, and
records its decisions beside the profile (`.copy.json`). It runs no tuning and writes no profile. The copy stage takes
no `--tokens` or `--prompt`; those belong to the lookup tuning. Without `--stage`, the copy is followed by the
combined tuning described below, until the load and Engram stages replace it.

`--copy_workers=N` (1 to 32, default 24) is the copy stage's own parameter: N worker threads, each taking one 64 MiB
chunk at a time through the source read, its SHA-256, the writes, the flush, the readback and the compare. The N
workers are one pool shared by all targets: with k targets a chunk costs 1 read + 1 hash + k writes + k readbacks. A
copy therefore runs at about N chunks over the sum of those stage times, until a drive's rate binds. The default of 24
measured fastest on the 9100 PRO host: 4.99 GB/s to one target against 3.62 with the earlier 4, and 4.55 against 2.77
to two (request 25b, 15 Nov 2026). The bytes and digests are the same at any count, and `--copy_workers=4` keeps the
earlier pace. The decisions record it (`"copy_workers"`), and each stage's `storage: copy stage` line reports its busy
seconds and its bytes per busy second.

`--copy_hash_pairs=1` (the default) hashes the source two chunks at a time: a worker keeps a chunk whose readbacks
matched until it has a second of the same length, then computes both SHA-256 digests in one interleaved pass (the
x86-64 SHA extensions run two streams in about the time of one). The digests, and so the manifest, are the same; the
copy keeps one more 64 MiB buffer a worker. It measured faster on the 9100 PRO host: 96 GiB to one target at 24
workers copied at 4.93 and 5.02 GB/s paired against 4.84 and 5.00 with one hash a chunk (request 27, 15 Nov 2026).
`--copy_hash_pairs=0` hashes one chunk at a time, the earlier default. The load stage refuses it by name.

`--stage=load` runs the load stage alone. It tunes the TP host arena's read, the part of a TP load that fills the
host memory with the n-gram and front-end tables, through the loader itself (`tp_host_load_run`):

```sh
anemone autotune /srv/model.anemone --stage=load --profile=/srv/runtime/storage-profile.json
```

- **Copies.** The copies are the ones serving would read: the source, then the receipted replicas that the replica
  search finds and that open beside it (`replicas_reader_open`). `--replica_external=0|1` is the search's policy.
- **Parameters.** Two parameters are tuned: the host arena's readers a copy (a cap on the host arena's own reads, 0 =
  no cap, the default) and the piece (`--load_piece_mib`, 16 MiB by default). The stage measures the host arena
  alone, so serving applies the section's readers to the host arena's reads only; the cards' reads beside it are not
  capped by them. `--tp_load_reads_per_copy` keeps its meaning, a cap on the whole TP load's reads, and when given
  it replaces the section's readers.
- **Points.** Every point is a real host load on serving's 16 host threads, with stealing between the copies and the
  chunk digests on the reading threads. It reads into an arena mapped afresh for the point in 2 MiB pages, as
  serving maps its own, so each point pays the first touch.
  - A point reads a window of the host-only extents, `--window_gib` (default 64) of whole extents. The window's
    start moves on by one window each point, so successive points read different regions.
  - The grid crosses readers 0, 1, 2, 3, 4, 6 and 8 with pieces of 4, 8, 16, 32 and 64 MiB.
- **Choice.** After the grid, the fastest arm other than serving's default (no cap, 16 MiB) runs A D D A against the
  default at twice the window.
  - Before its first point, one unmeasured arena of the A D D A's window is mapped and touched page by page, so that
    point is not the first touch of an arena that size (R26: a cold first point read 8.9 GB/s against 35 to 48). The
    warm-up's bytes, wall and CPU times are recorded as `"warmup"`, never ranked.
  - The higher mean wins, even inside the noise: a faster option that reads the same bytes becomes the default.
  - Only means that are equal at the reported precision (GB/s to three decimals) go to the arm with fewer readers
    (no cap counts as the most), then to the smaller piece.
  - The spread (the larger arm's A D D A range over its mean) and the margin over the default are recorded beside the
    choice. When the default wins, the section holds the default's values. A spread over 25 % is flagged
    (`"spread_over_bound":1` and the selection line), not refused: look at the points before `--qualify`.
- **Temperatures.** Every point records each NVMe drive's composite temperature before and after it.

Progress is kept in `<profile>.load-state` after every point. A lease (`--seconds`) or `--points_per_run=N` that
ends first exits 75, and the same command resumes. `--plan` prints the copies, the host bytes and the points without
reading. The stage refuses the copy stage's options (`--target`, `--copy_workers`, `--copy_hash_pairs`, `--manifest`,
`--verify`, `--replica_floor_percent`) and the lookup tuning's (`--tokens`, `--prompt`) by name.

**The load section.** The stage writes a schema 3 profile: `{"schema":3,"source":...,"load":{...}}`.
- **Other members stay.** A later stage's member, such as `"engram"`, stays byte for byte; writing the section again
  replaces only `"load"`.
- **Refusals.** A schema 2 profile (the combined tuning's) or another source's profile is refused by name before
  any point is measured (`--plan` prints `"valid":false` for it), and again when the section is written. While the
  combined tuning's schema 2 profile exists (at the default path too), give the stage its own `--profile`.
- **Contents.**
  - The pack's header digest and size.
  - The copies' device keys, paths and filesystems.
  - `"tp"`: `readers_per_copy` and `piece_bytes`.
  - Every point, with its rate, CPU time, per-copy bytes and temperatures.
  - The A D D A's warm-up: its bytes, wall and CPU times.
  - The selection: the winner, the challenger, the default, the A D D A rates, the spread with its bound and flag, the
    margin, whether it was an exact tie and whether the default holds.
  - `"pp"`: the PP loader's values are measured by real loads on the cards, not by this stage, so its defaults hold.
    R24 (15 Nov 2026, PP5 serve loads A D D A) set its readers a copy to 4: 16.442 and 16.399 s against 16.854 and
    17.207 s with 2, the earlier default (`--reader_threads_per_copy=2`); its piece stays 16 MiB.

The section is a candidate: `"enabled":0,"qualification":"candidate"`. `anemone autotune PACK --stage=load --qualify`
enables it once it names this pack and these copies.

A TP load applies an enabled section only while its pack digest and size, its copies' device keys in their order,
and the host threads and copy split it was measured with (16 and steal: `--tp_load_host_threads` and
`--tp_load_copy_split` at their defaults) match the load's. For an EXL3 map the read mode is part of the shape too: the
stage reads as `--map_reads` auto does (1 for a lone map, 3 with replicas), so an explicit `--map_reads` of another
value does not match (`load_section_stale_shape`), and auto or the same value does. Each parameter is taken from its knob when one is given, then from the section, then from
the default. The startup line `TP load read: R readers a copy (knob|profile|default), piece P bytes (...)` says which
applied (`profile = default` when the section's value is the default's), and when the section was not applied, why: `missing`, `not_a_stage_profile`, `no_load_section`,
`load_section_malformed`, `load_section_candidate`, `load_section_stale_pack`, `load_section_stale_copies` or
`load_section_stale_shape`.
Without a profile the TP load reads exactly as before.

Token files are little-endian uint32 IDs from the real serving protocols. Repeat
`--tokens=FILE --prompt=N` for each independent real prompt/continuation; inputs
are hashed and processed through the production n-gram hash and gather code.
No uniform/synthetic offset generator is used. Prompt prefixes are replayed table-first in
chunks of at most1024 tokens, reserving the final prompt token for the first
decode step, as in the engine. Decode uses at most64 observed single-token steps
per trace at C1/C2/C4/C8; short continuations stop without wrapping/repeating.
Rejected DSpark candidate coverage and end-to-end decode impact still require
native traces/gates. Images/dead-token inputs are outside this initial command.

`--plan` performs read-only device/argument planning without copying, tuning,
creating directories or writing a profile. For each selected target it also
prints what a run would do with the partial beside it (`storage: plan checkpoint
<partial> none|engine|adoptable|foreign|held|unknown bytes=N allocated=N: <action>`)
and counts the kinds in its JSON line. It checks in the run's order: the directory,
the profile's folder (it must be this user's alone: owned by this user and writable
by no group or other user; a missing one the run creates 0700), the copy lock (and
whether the run may create or write it: a read-only filesystem
or an unwritable directory is refused there, and nothing is removed), a final beside
the partial, the earlier copier's lock, the journal, the space admission (also for a
directory the run would create) and the adoption proof, so a refusal that comes
before the proof is named as such, never as adoptable. It holds no lock (each one is probed
shared and released at once), creates nothing and writes no journal; an adoption
proof reads the leading chunks with O_NOATIME. A directory it cannot open is
`unknown`, with the error.

Use repeated `--target=/mount/anemone_engram_rnd/name/model.anemone` to choose
explicit full-copy destinations, or omit targets for device discovery. Existing
verified finals are reused; an existing partial is read and hashed chunk by
chunk, including chunks whose progress bits are set. Corrupt partial chunks are
repaired; corrupt finalized files are excluded without overwrite. The engine
can reuse a complete existing schema1 source manifest with `--manifest=FILE`;
otherwise it builds the manifest during the same source fan-out pass.

Copy preparation is engine C, with full preallocation before any payload write,
statx-verified direct-I/O alignment, four bounded chunk workers,64MiB hash chunks
and8MiB io_uring pieces. Each source chunk is read once and shared by every target
that needs it; simultaneous writes have bounded backpressure from slow devices.
In-process SHA-256 (SHA-NI on x86-64, plain C rounds elsewhere), target fdatasync and
readback, durable chunk bitmaps and non-overwriting rename protect publication.
A written chunk is read back only once 8 GiB more were written to its target after
it and an fdatasync covering it returned, so the read comes from the flash, not the
drive's DRAM cache (8 GB on the 9100 PRO 8 TB, 4 GB on the 9100 PRO 4 TB and 990
PRO). The readback is compared byte for byte with the chunk's source buffer, which
is hashed against the manifest only after that comparison: one hash a chunk, and a
buffer that changed in memory at any point fails one of the two. A chunk enters
the journal only once a readback after its lag verified it (a read inside the lag
proves no flash, so it verifies the chunk for this run's publication but claims
nothing); the journal is saved once a GiB of claimed chunks
and at the run's end, not once a chunk. The retained source buffers cost the lag
plus one 64 MiB chunk a worker and one more, about 8.3 GiB for the whole run
whatever the number of targets; the copy is refused as `insufficient_memory` unless
MemAvailable exceeds that by 8 GiB. A run that would create a target (neither its
final nor its partial exists) checks this before it creates anything, so a refused
first run leaves no partial; a resumed run checks it once its targets are open. A
run that writes no target (every included one is a final it only verifies) retains
nothing: one buffer a worker and one more, and no memory refusal. A failed target
holds no chunk's lag. The run's last 8 GiB (and, on a resume that writes a target
only in places, a chunk read back when every buffer waits and none has passed its
lag) are read back inside the lag, after the flush. At a stop (the run's deadline
or a stop request) no chunk is written any more, and the written chunks still
waiting are read back at once, inside the lag and counted, for at most 10 s, so that
a bad write is found by this run; they stay unclaimed. A worker whose readback
outlasts that limit ends the lease as timed out and resumable, not as an I/O
failure. The next run reads every chunk of the
partial again before it publishes, claimed or not, and publication rests on those
reads. A chunk the journal claimed was read back once already after its lag, so
its read verifies it again. A chunk the journal had not claimed (one the stop read
back inside the lag or never read back, or one written before a crash) may have
been written in the stopping run's last 8 GiB, so its read may come from the
drive's cache: it is counted inside the lag. A chunk with no manifest hash yet (a first copy's, whose
hash comes after its readback) is written again rather than read. The copy's report
line `readbacks=N inside_lag=M` counts all of these. io_uring refusal uses threaded
direct preadv/pwritev, with a visible mechanism line. Target containers owned by
the runtime are secured to0700; no mount or filesystem configuration changes.
FIEMAP after publication records count/largest extent and warns on fragmentation.
Per-device written/verified bytes, elapsed time, GB/s and resumability are visible.
Shared/reflink/encoded or unknown allocations receive no space credit on resume.
No borrowed-source cache advice or global cache drop is used.

The command checkpoints across bounded leases: exit75 means repeat the SAME
command with the SAME profile/manifest/targets. Only fully verified files enter
tuning. A partial stays only with explicit durable resume state; space/policy
refusals remove its owned partial. Each invocation rechecks identities and floor.
Progress is stored beside the profile (`.copy.json`, `.tune-state`, and the source
manifest) and beside each replica (`.copy-state`, `.copy-lock`). Do not remove
these during an active or resumable run. Copy time is separate from tuning and
ordinary startup/lookup measurements. All copies/benchmarks need the operator's
CPU/disk window with GPU idle and no model load or timed GPU job.

This initial CPU search compares page-locked16MiB sequential reads with1/2/4
threads per device, production direct/fixed-file io_uring gathers, solo devices
and, for every set of two or more copies (with more than three copies, the full set
only), lookup shares that are equal, proportional to each copy's solo speed, or to
its square. Load points read every copy, so they run on the full set only, and
their readers steal: each copy's readers take the next 16 MiB chunk from one
cursor, as the loader's do, so a faster copy reads more. Every load point reads
the same 8 GiB (less of a smaller pack) and is ranked by time per byte. The
profile's load shares are the winner's chunks per copy, recorded with
`"share":"steal"`; the loader applies them only with `--load_tuned_weights=1`
(a static split, for comparison), and otherwise steals. Each lookup
point runs with the bounded cache and again uncached (the block cache bypassed;
rows in the RAM prefix still come from memory); only cached points can win the
candidate's lookup. It records
refused points (including memlock limits), per-step p50/p99, prefill chunk costs,
the rows gathered and wall time per stream level (rows per second),
all decisions and exact reference-hash agreement. Unique-row ratio and exact
block LRU reuse-distance histogram on the natural concatenated trace choose a
bounded conditional-p90 cache budget (64..512MiB). In a profile, a zero load share
keeps a retained copy for lookup only (it is left out of the load list); this
tuner's load winner reads every copy, and its lookup may set a slow copy's share
to 0.
Broader mechanism/QD/stripe/NUMA search and cross-load/native effects remain gates.

CPU results are saved as an **inactive candidate** (`enabled:0`) with both action
sections, hardware identities, source manifest and every measured/refused point.
This protects the unchanged defaults until native startup-to-serving, paired
TTFT/decode/prefill/concurrency and numerical/state qualification. The runtime
owner activates/installs only the qualified profile; every later start, including
after reboot, reloads it automatically. The recommendation does not authorize
implicit tuning at boot or an unqualified speed claim. The Python storage module
is retained for tiny host fixtures/qualification helpers; its production copy,
manifest and tune CLI routes are removed.

Sequential startup loading and random Engram lookup are separate actions. They
share device discovery, physical identities and verified copies, and have
independent measured weights and settings. Without an enabled qualified profile,
ordinary startup scheduling and all-RAM Engram remain the defaults. No axis
selects an unqualified storage profile.

Default candidates are internal local nonrotational NVMe/SSDs. Discovery resolves
partitions and dm/md slaves to physical disks/controllers; overlapping sets and
several mounts of one disk cannot create extra bandwidth. Network filesystems
and transports, FUSE, tmpfs/ramfs/overlay, loop/virtual devices and unknown
physical topology are excluded. The checks use filesystem/mount information,
sysfs topology/flags and udev ID_BUS when available. SD/MMC, removable media and
USB enclosures require `--replica_external=1`. This explicit override permits
external media, including a forced slow external disk, but never network/virtual
storage. Internal rotating disks stay excluded. Inclusion still requires stable
identity, space, verification and explicit performance qualification; slow media
can receive a small or zero measured share.

Before each copy, available filesystem bytes must cover the remaining allocation
plus `--replica_floor_percent` of total filesystem bytes (default20, range0..99).
Already allocated resumable chunks are credited, and an existing finalized file
needs no new allocation. The tool checks the floor during copying and checks
media/identity/space again immediately before atomic publication. A device that
does not fit is automatically excluded, with one product reason line; processing
continues for the remaining candidates. Only the process holding the target's
`.copy-lock` removes a partial, only its own (a `.copy-state` journal whose pack,
hardware key, geometry and inode match, or a partial it created; never an
adopted one), and only for a space, read-only or explicit media-policy refusal, including a
mid-copy shortfall. Every other refusal (a describe, identity or I/O failure, any
transient) keeps the partial and its journal and prints one line naming the file,
its size and the reason. An engine checkpoint beside a final is removed once the
final verifies every byte, and its bytes count as available meanwhile.

A partial with no matching journal is foreign, such as one an earlier copier left.
The engine never removes or changes it. It adopts one only on proof: an owned
regular file of the pack's size, not written for 15 minutes, whose leading chunks
(the first at least, up to 1 GiB) hash to the source manifest and which does not
change while they are read. Adoption writes the journal (schema 2 records the
adoption); every chunk is then verified again as the copy resumes, and an adopted
checkpoint is never removed: it only completes into the final. For every target not
yet final the engine also takes the earlier copier's lock (`.<name>.copy.lock`, as
`tools/storage/manager.py` takes it) right after its own and holds it to the end of
the run, so the two never work on one partial: while that copier holds it the
target is refused as `copy_in_progress`, and nothing is adopted, resumed or
created. A lock is tried three times 1 ms apart before another holder is assumed. Any other
foreign partial refuses its target as `foreign_partial_kept`, with a line saying it
is kept and what to do: move or remove it once no copier holds it. A target whose
`<name>.copy-lock` another copy holds is refused as `copy_in_progress`, and its
partial is left unexamined; a refusal that comes before the proof (space, for
example) says the partial was not examined for adoption, with no removal advice. A
fresh partial is created unnamed and linked at its name only once its journal is
saved, so a killed run leaves no partial without a journal; it counts as the run's
own only once linked, so a failed link never removes a file another writer put at
that name, and a failed directory sync after the link keeps the journaled
checkpoint (`copy_device_io_failure`).

After a final's full verification (its copy, or a later full read) the engine writes a verification receipt beside
it, `<name>.verified`: the device's hardware key, and the inode, size, mtime and ctime the verification started
from, a digest of the manifest's chunk hashes and the pack identity, written atomically. A later run whose final
still matches its receipt reads only a few random chunks of it (a spot check) instead of every byte, so a resumed
autotune spends its lease on tuning; any change of hardware key, inode, size, mtime, ctime, manifest or pack forces
a full read, and `--verify=full` always reads every byte. The receipt and the copy journal name the device by its
hardware key, not its device number, so a reboot that renumbers NVMe partitions keeps both. A final that changes
while a run verifies or spot-checks it is refused for that run and gets no receipt. A final that fails its
verification loses its receipt at once (unlinked, its directory synced), so a crash later in the run cannot leave
it; a final refused for any other reason (identity, space, policy) keeps it. The decision records give a final's
`receipt_trusted_bytes` beside its `verified_bytes`. The trade-off: bytes that change with no metadata change (a
device's silent corruption) are caught only when a spot chunk lands on them or by `--verify=full`.

Serving never trusts a replica's bytes on its receipt: every read is checked against the pack's own digests and
falls back to another copy, so a receipt only spares autotune's verification pass.
- Turbo load (PP): each extent that arrives, on a card or in host memory, is digested and compared with the pack's
  extent digest (source/loader.c:443); an extent that differs is read again alone from each live copy in turn until
  its digest is the pack's, else the load fails with "has the digest of the pack on no copy" (source/loader.c:461,
  496).
- Turbo load (TP): each extent is digested before it is staged (source/tp_load.c:374); a wrong digest faults that
  copy and the next is read (source/tp_load.c:379), and no matching copy fails the load (source/tp_load.c:381).
- Engram on disk: at open each table is read from one copy and accepted only if its digest is the pack's table digest,
  else the next copy is read (source/engram_disk.c:173); the per-block digests come from that verified read. Every
  later block read is compared with its block digest and falls back to the next copy, healthy copies first
  (source/engram_disk.c:354); the batched io_uring path checks each block the same way and repairs a mismatch through
  that read (source/engram_disk.c:475), and the bounded cache holds only blocks that passed (source/engram_disk.c:493,
  559). The RAM prefix is copied from the verified open read. Each copy also
prints where its time went: one `storage: copy stage <name>` line per stage (source read and hash, target read and
hash, write, flush, readback read and hash, journal) with its busy seconds summed over the workers and its bytes per
busy second; `tests/replica_copy_bench_test` runs the copy on real folders for these numbers. Interruptions may
retain a resumable partial;
partial files never enter a serving profile. No floor is lowered automatically.
New learned copies belong only in new `anemone_engram_rnd/` directories.

Source manifests use schema1 and64MiB SHA256 chunks. Resuming hashes durable
partial chunks before reuse, verifies source bytes, writes/fsyncs and reads back.
Publication uses Linux non-overwriting rename, directory fsync and an atomic
complete receipt. A corrupt finalized replica is excluded without repair or
overwrite. Cache advice applies only to this task's destination files, never
borrowed source packs; no system cache is dropped. Copy/tune commands require
an explicit queue-owned window and quiet-marker directory. They never acquire
the GPU lock; schedule copies with idle I/O priority outside model loading and
timed work (some device schedulers do not implement that priority).

Every normal load/generate/serve start checks an owned profile before model
loading. Default location: `$XDG_CONFIG_HOME/anemone/storage-profile.json`, or
`$HOME/.config/anemone/storage-profile.json`. An installation can choose a fixed
adjacent location with `--storage_profile=/absolute/path/storage-profile.json`
or `ANEMONE_STORAGE_PROFILE`; the CLI wins and `off` disables reload. With
loading enabled and no valid profile, startup prints the `anemone autotune`
hint; an explicit `off` is a choice and prints no hint. Explicit
replica-policy flags bypass the saved profile. Explicit settings in another
action's family override that action. No start benchmarks, copies or retunes.

The profile uses schema2; source manifests remain schema1. Required shape:

```json
{
  "schema": 2, "enabled": 0, "created_unix": 0,
  "source": "/srv/model/source.anemone",
  "manifest": "/srv/model/source-manifest.json",
  "manifest_sha256": "<64 lowercase hexadecimal characters>",
  "replica_policy": {"external": 0, "floor_percent": 20},
  "decisions": [
    {"path": "/srv/anemone_engram_rnd/replicas/model.anemone",
     "key": "<hardware SHA256>", "included": 1, "reason": "eligible",
     "mount": "/srv", "model": "example SSD", "transport": "nvme",
     "filesystem": "ext4", "physical_devices": ["0000000000000001"]}
  ],
  "devices": [
    {"path": "/srv/anemone_engram_rnd/replicas/model.anemone",
     "key": "<hardware SHA256>", "load_weight": 3, "engram_weight": 7}
  ],
  "turbo_load": {"enabled": 0, "threads": 1, "mechanism": "direct_pread", "share": "steal"},
  "turbo_engram": {"enabled": 0, "ram_percent": 50, "cache_mib": 256,
                   "io": 1, "mechanism": "uring_fixed_direct",
                   "step_io": {"selected": "hybrid", "threads_per_copy": 2, "spin_us": 200, "pin": 1,
                               "applies": "serving PP5 or TP4_PRO from disk tables", "evidence": "..."}}
}
```

The Engram section's `step_io` records the stage's selection and the serving evidence it rests on (step I/O was
measured in real serving, not by the tuner). Startup does not read it: serving PP5 or TP4_PRO from disk tables uses
step I/O by default (`--engram_step_io`).

This inactive example is not a tuned configuration. Weights are relative request
shares from separately measured aggregate winners, never assumed equal or derived
only from solo rates. Zero can remove a device from one action; no positive
surviving weight disables that profile action. For startup loading, a profile
supplies the copies and the readers per copy, and by default the readers of every
copy steal the next piece, so a faster drive reads more: the load weights choose a
piece's copy only with `--load_tuned_weights=1`, a static split kept for
comparison (with equal weights the slowest drive paces the whole load). Every active device must match
an included decision; excluded candidates remain in `decisions` with their reason.
Reasons are fixed named codes, not arbitrary text. Missing fields or schema1
profiles safely fall back and need explicit operator migration/qualification.

The hardware key covers physical topology, model/serial/WWN/firmware, relevant
PCIe/USB/card identity, filesystem UUID, mount/type/options and kernel. Reload
checks policy, current identity, free floor and every replica SHA256 chunk.
Removed, changed, short, corrupt or duplicate devices drop individually with
visible reasons; the remaining verified devices retain their relative shares.
A duplicate physical device (two paths to one disk, or one replica recorded
twice with identical decisions) keeps the first verified path and excludes the
other with a line naming both paths and the shared physical identity.
An EXL3 map (`.exl3map`) lies on the drive of its largest source file, not on
the drive of the map file. That drive is the one used for the duplicate test,
for its key and in the storage lines, and so is the drive a replica search sets
the source on. An EXL3 profile's source is its map. Its identity and every
manifest chunk are those of the logical pack, read through the map reader.
The map is never a disk lookup source, because the disk store reads packs. A
profile whose only verified device is the map loads from it and keeps lookup
off: `storage profile: fallback engram_sources_none`. Only a
structurally invalid profile is refused: bad schema or fields, a digest
mismatch, a contradictory decision for one path, or a foreign source pack
identity.
Previously excluded candidates are reconsidered for diagnostics but never
silently added or benchmarked. No verified device gives safe defaults. A missing
original source can use a verified byte-identical replica; a changed existing
source cannot silently select an older model. Files remain immutable while used.
Full verification is ordinary startup cost and must fit the serving readiness
budget; one-time manifest/copy/tuning is excluded from startup comparisons.

Decisions use product stderr lines in every build, including release:

```text
storage: replica /mnt/test (fixture_ssd, usb, 0000000000000001) excluded: external_device_requires_override (use --replica_external=1)
storage: replica /mnt/test (fixture_ssd, nvme, 0000000000000001) included: verified_identity_and_bytes load_weight=3 lookup_weight=7
storage: replica /srv/b/model.anemone excluded: duplicate_physical_device (shares physical device 0000000000000001 with kept replica /srv/a/model.anemone)
```

The space reason also prints required bytes, floor percent and available bytes;
network refusal names the filesystem. Reload also prints fixed old/new reason codes when a decision changes.
An excluded network decision keeps its recorded reason at reload and its path is
never described again, and a path below an automount that has not fired is
refused before any filesystem call, so startup never touches a network mount. An
explicit `--target` below such an automount is refused as `automount_not_mounted`,
and nothing is created there.
A destination or mount point that is no UTF-8 is refused
(`destination_directory_policy`) and recorded with U+FFFD in its place; the rest
of the profile stays readable, and two such refused decisions that read alike do
not contradict each other. A source, `--profile` or `--manifest` path that is no
UTF-8 refuses autotune before any copy.
No prompt, request, credential or serial
is printed. Model-free `anemone storage-device PATH`, `storage-room PATH BYTES`,
`storage-scan [--replica_external=1]` and `storage-report PROFILE` expose the same
live policy and saved-profile revalidation without opening a model/CUDA context.
Report revalidation reads full replica hashes; schedule that I/O appropriately.
`storage-report PROFILE --metrics=1` emits the bounded info family documented in
[METRICS.md](../METRICS.md). The optional metrics server consumes an immutable
startup snapshot, with no disk scan or allocation on scrape.

The initial runtime subset is direct pread startup with thread count/shares,
and direct/fixed-file io_uring lookup with RAM share and bounded cache. Broader
per-device mechanism/flag/queue/stripe/affinity search and native optimization
remain explicit qualification work. Each plan point explicitly names its device keys. Changed or excluded devices remove only points
that depend on them; remaining devices can still be measured. Available-byte fluctuations do not
change a stable identity. The pinned-plan runner stores measured and
refused points, solo/aggregate/contention conditions, native gates and runners-up.
Unsupported or native-unqualified winners cannot be published. `publish` checks
both winners, exact configs/weights, every decision/identity and the manifest,
then validates with the engine and writes temp+fsync+rename+directory fsync.
An identical repeat is idempotent; replacement requires explicit `--retune`.

The runtime owner installs a qualified profile, launch change and guarded restart.
There is no in-place model-pointer replacement. Each later start, including after
reboot, reloads the saved profile; no implicit tune runs during boot. Numerical,
state and paired startup/TTFT/decode/prefill/concurrency gates remain required
before four-axis promotion.
