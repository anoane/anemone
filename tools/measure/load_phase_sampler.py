#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Splits today's model load into phases by watching it from outside.

Starts one command (the port loading the model and exiting), and four times a second records how many bytes each
drive has read and how much memory each card holds. From that timeline it reports: the time before the first
byte, the seconds in which bytes flowed and at what rate from each drive and into each card, the longest stalls,
and the time after the last byte. That tells which part of the 61 to 71 s is the drives, which the links, and which
is neither.

    load_phase_sampler.py --json <file> -- <command that loads the model and exits>

Measurement tool: not part of the engine.
"""
import argparse
import json
import subprocess
import sys
import time

SAMPLE_INTERVAL_SECONDS = 0.25
SECTOR_BYTES = 512                                  # /proc/diskstats counts sectors of 512 bytes whatever the drive uses
DISKSTATS_DEVICE_NAME_FIELD = 2
DISKSTATS_SECTORS_READ_FIELD = 5
BYTES_PER_MIB = 2 ** 20
BYTES_PER_GIB = 2 ** 30
ACTIVE_RATE_MIB_PER_SECOND = 50.0                   # below this a sample counts as "no bytes flowing"
STALLS_PRINTED = 5


def read_drive_bytes():
    drives = {}
    with open("/proc/diskstats") as handle:
        for line in handle:
            fields = line.split()
            name = fields[DISKSTATS_DEVICE_NAME_FIELD]
            if name.startswith("nvme") and "p" not in name[len("nvme"):]:          # whole drives, not partitions
                drives[name] = int(fields[DISKSTATS_SECTORS_READ_FIELD]) * SECTOR_BYTES
    return drives


def read_card_memory_bytes():
    output = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                            capture_output = True, text = True, check = True).stdout
    cards = {}
    for line in output.strip().splitlines():
        index, used_mib = [part.strip() for part in line.split(",")]
        cards[f"card {index}"] = int(used_mib) * BYTES_PER_MIB
    return cards


def summarize(samples, key, label):
    """Per source (a drive or a card): total bytes, active seconds, mean rate while active."""
    names = sorted(samples[0][key])
    for name in names:
        active_seconds, moved_while_active = 0.0, 0
        for earlier, later in zip(samples, samples[1:]):
            moved = later[key][name] - earlier[key][name]
            interval = later["time"] - earlier["time"]
            if moved / BYTES_PER_MIB / interval >= ACTIVE_RATE_MIB_PER_SECOND:
                active_seconds += interval
                moved_while_active += moved
        # the peak, not the last sample: a card's memory is back at its start once the command has exited
        total = max(sample[key][name] for sample in samples) - samples[0][key][name]
        if total <= 0:
            continue
        rate = moved_while_active / BYTES_PER_GIB / active_seconds if active_seconds else 0.0
        print(f"  {label} {name}: {total / BYTES_PER_GIB:7.1f} GiB in all, active {active_seconds:5.1f} s at {rate:5.2f} GiB/s")


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", default = None)
    parser.add_argument("command", nargs = argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    if not command:
        parser.error("a command is required after --")

    started = time.monotonic()
    process = subprocess.Popen(command)
    samples = []
    while True:
        samples.append({"time": time.monotonic() - started, "drives": read_drive_bytes(), "cards": read_card_memory_bytes()})
        if process.poll() is not None:
            break
        time.sleep(SAMPLE_INTERVAL_SECONDS)
    total_seconds = samples[-1]["time"]

    def flowing(earlier, later):
        interval = later["time"] - earlier["time"]
        drive_bytes = sum(later["drives"][name] - earlier["drives"][name] for name in later["drives"])
        card_bytes = sum(max(later["cards"][name] - earlier["cards"][name], 0) for name in later["cards"])
        return max(drive_bytes, card_bytes) / BYTES_PER_MIB / interval >= ACTIVE_RATE_MIB_PER_SECOND

    activity = [flowing(earlier, later) for earlier, later in zip(samples, samples[1:])]
    first_active = next((index for index, active in enumerate(activity) if active), None)
    last_active = next((index for index in range(len(activity) - 1, -1, -1) if activity[index]), None)
    print(f"command exit code {process.returncode}, {total_seconds:.1f} s in all, {len(samples)} samples")
    if first_active is None:
        print("no bytes flowed")
        return 1
    print(f"  before the first byte: {samples[first_active]['time']:.1f} s; after the last byte: {total_seconds - samples[last_active + 1]['time']:.1f} s")
    stalls, stall_start = [], None
    for index in range(first_active, last_active + 1):
        if not activity[index] and stall_start is None:
            stall_start = index
        if (activity[index] or index == last_active) and stall_start is not None:
            stalls.append((samples[index]["time"] - samples[stall_start]["time"], samples[stall_start]["time"]))
            stall_start = None
    print(f"  stalls between first and last byte: {sum(length for length, _ in stalls):.1f} s in {len(stalls)} gaps; longest: "
          + ", ".join(f"{length:.1f} s at {at:.0f} s" for length, at in sorted(stalls, reverse = True)[:STALLS_PRINTED]))
    summarize(samples, "drives", "drive")
    summarize(samples, "cards", "into")
    if arguments.json:
        with open(arguments.json, "w") as handle:
            json.dump({"total_seconds": total_seconds, "exit_code": process.returncode, "samples": samples}, handle)
    return 0 if process.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
