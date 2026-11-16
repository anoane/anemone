#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CPU test fixtures and qualification helpers. Production replica preparation is engine-native autotune."""
import argparse
import contextlib
import ctypes
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time
import signal
import resource
import sys
import errno

SCHEMA = 2
MANIFEST_SCHEMA = 1
DEFAULT_FLOOR_PERCENT = 20
CHUNK = 64 << 20
LIMIT = 4 << 20
MOST_CHUNKS = 16384
FLOOR_DENOMINATOR = 5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def read_json(path):
    with open(path, 'rb') as stream:
        data = stream.read(LIMIT + 1)
    require(len(data) <= LIMIT, 'document too large')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate key')
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique, parse_constant=lambda _: require(False, 'nonfinite JSON'))


def digest(value):
    return hashlib.sha256(value).hexdigest()


def rename_no_replace(source, destination):
    """Linux atomic publication without overwriting a file that appeared during copying."""
    rename = getattr(ctypes.CDLL(None, use_errno=True), 'renameat2', None)
    require(rename is not None, 'renameat2 unavailable; partial replica retained')
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    at_current_directory, no_replace = -100, 1
    if rename(at_current_directory, os.fsencode(source), at_current_directory, os.fsencode(destination), no_replace):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def atomic_json(path, value, checkpoint=lambda _: None):
    """Old or complete new document after a crash; directory fsync commits the rename."""
    path = Path(path)
    require(path.parent.is_dir(), 'parent directory must already exist')
    data = canonical(value)
    require(len(data) <= LIMIT, 'document too large')
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        checkpoint('file_fsynced')
        os.replace(temporary, path)
        checkpoint('renamed')
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        checkpoint('directory_fsynced')
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def command_json(arguments, timeout=15):
    return json.loads(subprocess.check_output(arguments, timeout=timeout))


def hardware(path, external=False):
    """Stable keys come from the engine's CPU-only identity command, not a second implementation."""
    engine = os.environ.get('ANEMONE_STORAGE_ENGINE')
    require(engine is not None and Path(engine).is_absolute(), 'ANEMONE_STORAGE_ENGINE must name the pinned engine')
    return command_json([engine, 'storage-device', str(path), '--replica_external=' + str(int(external))])


def copy_room(path, bytes_needed, external=False, floor_percent=DEFAULT_FLOOR_PERCENT):
    require(type(floor_percent) is int and 0 <= floor_percent < 100, 'invalid free-space floor')
    require(type(external) is bool and type(bytes_needed) is int and 0 <= bytes_needed < 2**64, 'invalid replica policy')
    engine = os.environ['ANEMONE_STORAGE_ENGINE']
    return command_json([engine, 'storage-room', str(path), str(bytes_needed),
                         '--replica_external=' + str(int(external)), '--replica_floor_percent=' + str(floor_percent)])


REPLICA_REASONS = frozenset([
    'eligible', 'verified_identity_and_bytes', 'network_filesystem_or_transport', 'virtual_filesystem_or_device',
    'unknown_physical_topology', 'external_device_requires_override', 'rotational_device', 'read_only_filesystem',
    'invalid_space_policy', 'insufficient_space', 'invalid_device', 'device_disappeared_or_unknown',
    'hardware_identity_unavailable', 'hardware_identity_mismatch_retune_required', 'duplicate_physical_device',
    'zero_tuned_weight', 'replica_missing_short_or_unverified', 'previously_excluded_retune_required',
    'destination_directory_policy', 'unverified_final_exists', 'verification_failed', 'device_limit',
    'copy_io_error', 'copy_device_io_failure', 'finalized_replica_unverified', 'action_disabled', 'invalid_decision',
    'automount_not_mounted', 'foreign_partial_kept', 'copy_in_progress'])


def display_text(value):
    return ''.join(c if ' ' <= c <= '~' and c not in ('"', chr(92)) else '_' for c in str(value))


def product_line(device, reason, bytes_needed=0, floor_percent=DEFAULT_FLOOR_PERCENT,
                 load_weight=0, lookup_weight=0):
    reason = reason if reason in REPLICA_REASONS else 'invalid_decision'
    device = device or {}
    included = reason in ('eligible', 'verified_identity_and_bytes')
    line = 'storage: replica {} ({}, {}, {}) {}: {}'.format(
        display_text(device.get('mount') or 'unavailable'), display_text(device.get('model') or 'unknown'),
        display_text(device.get('transport') or 'unknown'), display_text(device.get('physical') or '0000000000000000'),
        'included' if included else 'excluded', reason)
    if included:
        line += ' load_weight={} lookup_weight={}'.format(load_weight, lookup_weight)
    elif reason == 'insufficient_space':
        line += ' needs_bytes={} floor_percent={} available_bytes={}'.format(
            bytes_needed, floor_percent, device.get('available_bytes', 0))
    elif reason == 'external_device_requires_override':
        line += ' (use --replica_external=1)'
    elif reason == 'network_filesystem_or_transport':
        line += ' filesystem=' + display_text(device.get('filesystem') or 'unknown')
    return line + chr(10)


def decision(path, included, reason, device=None, bytes_needed=0, floor_percent=DEFAULT_FLOOR_PERCENT):
    reason = reason if reason in REPLICA_REASONS else 'invalid_decision'
    included = bool(included) and reason in ('eligible', 'verified_identity_and_bytes')
    device = device or {}
    result = dict(path=str(path), included=int(included), reason=reason,
                  key=device.get('key', ''), physical_devices=device.get('physical_devices', []),
                  **{key: device.get(key, '') for key in ['mount', 'model', 'transport', 'filesystem', 'physical']})
    sys.stderr.write(product_line(device, reason, bytes_needed, floor_percent))
    sys.stderr.flush()
    return result


def existing_parent(path):
    parent = Path(path).parent
    for _ in range(128):
        if parent.exists():
            return parent.resolve(strict=True)
        parent = parent.parent
    raise ValueError('no existing ancestor')


def select_devices(candidates, bytes_needed, external=False, floor_percent=DEFAULT_FLOOR_PERCENT, existing_physical=(), resume=False):
    """One shared selection for both actions. Each rejected candidate leaves the remaining set usable."""
    require(type(candidates) is list and len(candidates) <= 64, 'candidate limit')
    result, physical = [], set(existing_physical)
    for candidate in candidates:
        path = Path(candidate['path'])
        require(path.is_absolute(), 'absolute candidate path required')
        # Refuse network mounts before Python stat/exists can wait on a remote filesystem.
        probe = hardware(path, external)
        if probe.get('reason') in ('network_filesystem_or_transport', 'virtual_filesystem_or_device',
                                   'external_device_requires_override', 'rotational_device'):
            result.append(decision(path, False, probe['reason'], probe, bytes_needed, floor_percent))
            continue
        needed = bytes_needed
        if resume:
            target = path if path.exists() else path.with_name(path.name + '.partial')
            try:
                info = target.lstat()
                if stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and info.st_nlink == 1:
                    needed = 0 if target == path else max(0, bytes_needed - info.st_blocks * 512)
            except FileNotFoundError:
                pass
        found = copy_room(existing_parent(path), needed, external, floor_percent)
        refused = None if found.get('admitted') else found.get('reason', 'device_disappeared_or_unknown')
        if refused is None and candidate.get('key') is not None and candidate['key'] != found['key']:
            refused = 'hardware_identity_mismatch_retune_required'
        leaves = set(found.get('physical_devices', []))
        if refused is None and (not leaves or leaves & physical):
            refused = 'duplicate_physical_device' if leaves else 'unknown_physical_topology'
        if refused is None:
            physical.update(leaves)
        result.append(decision(path, refused is None, refused or 'eligible', found, needed, floor_percent))
    return dict(schema=SCHEMA, replica_policy=dict(external=int(external), floor_percent=floor_percent),
                decisions=result, devices=[d for d in result if d['included']])


def remove_owned_partial(path):
    """Refusal cleanup never unlinks a symlink, alias, foreign file or finalized replica."""
    path = Path(path)
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and info.st_nlink == 1,
            'partial is not an owned regular file without aliases')
    path.unlink()
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def current_floor_refused(parent, floor_percent):
    space = os.statvfs(parent)
    total = space.f_blocks * space.f_frsize
    floor = (total * floor_percent + 99) // 100
    return space.f_bavail * space.f_frsize < floor


def owner_window():
    require(bool(os.environ.get('ENGRAM_OWNER_WINDOW')), 'queue-owned window required')
    quiet_directory = Path(os.environ.get('ANEMONE_STORAGE_QUIET_DIRECTORY', ''))
    require(quiet_directory.is_absolute() and quiet_directory.is_dir(), 'explicit quiet-marker directory required')
    require(not list(quiet_directory.glob('CPU_QUIET*')), 'CPU_QUIET marker present')


def regular(descriptor):
    info = os.fstat(descriptor)
    require(stat.S_ISREG(info.st_mode), 'regular file required')
    return info


def read_exact(descriptor, offset, count):
    data = bytearray()
    while len(data) < count:
        part = os.pread(descriptor, count - len(data), offset + len(data))
        require(part, 'short read')
        data.extend(part)
    return data


def source_manifest(source, output, pack_id, chunk_bytes=CHUNK):
    owner_window()
    require(4096 <= chunk_bytes <= CHUNK and chunk_bytes % 4096 == 0, 'invalid chunk size')
    descriptor = os.open(source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = regular(descriptor)
        require(0 < before.st_size <= chunk_bytes * MOST_CHUNKS, 'source too large')
        chunks = []
        for offset in range(0, before.st_size, chunk_bytes):
            owner_window()
            length = min(chunk_bytes, before.st_size - offset)
            chunks.append(digest(read_exact(descriptor, offset, length)))
        after = os.fstat(descriptor)
        require((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) ==
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns),
                'source changed during hashing')
        result = dict(schema=MANIFEST_SCHEMA, pack_id=pack_id, source=str(Path(source).resolve()),
                      bytes=before.st_size, chunk_bytes=chunk_bytes, chunks=chunks)
        atomic_json(output, result)
        return result
    finally:
        os.close(descriptor)


def check_source(value):
    require(value.get('schema') == MANIFEST_SCHEMA, 'manifest schema')
    size, chunk = value['bytes'], value['chunk_bytes']
    require(type(size) is int and type(chunk) is int and 0 < size <= CHUNK * MOST_CHUNKS,
            'invalid size')
    require(4096 <= chunk <= CHUNK and chunk % 4096 == 0, 'invalid chunk size')
    require(len(value['chunks']) == (size + chunk - 1) // chunk <= MOST_CHUNKS, 'chunk count')
    for item in value['chunks']:
        require(type(item) is str and len(item) == 64 and all(c in '0123456789abcdef' for c in item),
                'invalid SHA256')


def copy_replica(manifest_path, destination, receipt_path, expected_device,
                 external=False, floor_percent=DEFAULT_FLOOR_PERCENT):
    """Verified resumable copy. Policy/I/O refusal excludes this device and removes its owned partial."""
    owner_window()
    manifest = read_json(manifest_path)
    check_source(manifest)
    destination = Path(destination)
    probe = hardware(destination, external)
    if probe.get('reason') == 'network_filesystem_or_transport':
        return dict(schema=SCHEMA, complete=False, **decision(destination, False, probe['reason'], probe, manifest['bytes'], floor_percent))
    try:
        parent = destination.parent.resolve(strict=True)
    except OSError:
        return dict(schema=SCHEMA, complete=False, **decision(destination, False, 'device_disappeared_or_unknown', probe))
    require('anemone_engram_rnd' in parent.parts, 'new replicas belong under anemone_engram_rnd/')
    require(parent.stat().st_uid == os.geteuid(), 'destination directory must be owned by the copy runtime')
    require(Path(receipt_path).parent.resolve(strict=True) == parent, 'receipt must be beside its replica')
    partial = destination.with_name(destination.name + '.partial')
    import fcntl
    lock_fd = os.open(parent / ('.' + destination.name + '.copy.lock'),
                      os.O_WRONLY | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock_fd, 'wb') as lock:
        lock_info = os.fstat(lock.fileno())
        require(stat.S_ISREG(lock_info.st_mode) and lock_info.st_nlink == 1 and lock_info.st_uid == os.geteuid(),
                'copy lock must be an owned regular file without aliases')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_fd, target_fd = -1, -1
        published = False
        found = None

        def exclude(reason):
            nonlocal target_fd
            if target_fd >= 0:
                os.close(target_fd)
                target_fd = -1
            remove_owned_partial(partial)
            if published:
                remove_owned_partial(destination)
            return dict(schema=SCHEMA, complete=False, **decision(destination, False, reason, found, remaining, floor_percent))

        try:
            target = destination if destination.exists() else partial
            allocated = target.lstat().st_blocks * 512 if target.exists() else 0
            remaining = max(0, manifest['bytes'] - allocated)
            # Existing completed copies need no new allocation; their bytes are still fully verified below.
            if target == destination:
                remaining = 0
            found = copy_room(parent, remaining, external, floor_percent)
            if not found.get('admitted'):
                return exclude(found.get('reason', 'device_disappeared_or_unknown'))
            if found['key'] != expected_device:
                return exclude('hardware_identity_mismatch_retune_required')
            source_fd = os.open(manifest['source'], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            require(regular(source_fd).st_size == manifest['bytes'], 'source size changed')
            target_fd = os.open(target, (os.O_RDONLY if target == destination else os.O_RDWR | os.O_CREAT) |
                                os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
            info = regular(target_fd)
            require(info.st_dev == parent.stat().st_dev and info.st_nlink == 1 and info.st_uid == os.geteuid(),
                    'invalid destination')
            require((info.st_dev, info.st_ino) != (os.fstat(source_fd).st_dev, os.fstat(source_fd).st_ino),
                    'destination aliases source')
            require(info.st_size <= manifest['bytes'], 'destination size mismatch')
            if target == partial:
                os.posix_fallocate(target_fd, 0, manifest['bytes'])
            require(os.fstat(target_fd).st_size == manifest['bytes'], 'destination size mismatch')
            for index, expected in enumerate(manifest['chunks']):
                owner_window()
                if current_floor_refused(parent, floor_percent):
                    return exclude('insufficient_space')
                offset = index * manifest['chunk_bytes']
                length = min(manifest['chunk_bytes'], manifest['bytes'] - offset)
                if digest(read_exact(target_fd, offset, length)) != expected:
                    if target == destination:
                        return exclude('finalized_replica_unverified')
                    data = read_exact(source_fd, offset, length)
                    require(digest(data) == expected, 'source chunk differs from manifest')
                    written = 0
                    while written < length:
                        count = os.pwrite(target_fd, memoryview(data)[written:], offset + written)
                        require(count > 0, 'short write')
                        written += count
                    os.fdatasync(target_fd)
                    os.posix_fadvise(target_fd, offset, length, os.POSIX_FADV_DONTNEED)
                    require(digest(read_exact(target_fd, offset, length)) == expected, 'readback mismatch')
                os.posix_fadvise(target_fd, offset, length, os.POSIX_FADV_DONTNEED)
            os.fsync(target_fd)
            # This check is immediately before atomic file publication, after allocation and verification.
            found = copy_room(parent, 0, external, floor_percent)
            if not found.get('admitted'):
                return exclude(found.get('reason', 'device_disappeared_or_unknown'))
            if found['key'] != expected_device:
                return exclude('hardware_identity_mismatch_retune_required')
            if target == partial:
                rename_no_replace(partial, destination)
                published = True
            directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            if current_floor_refused(parent, floor_percent):
                return exclude('insufficient_space')
            receipt = dict(schema=SCHEMA, path=str(destination), device_key=expected_device,
                           source_manifest=str(Path(manifest_path).resolve()),
                           source_manifest_sha256=digest(canonical(manifest)), complete=True,
                           replica_policy=dict(external=int(external), floor_percent=floor_percent),
                           decision=dict(path=str(destination), included=1, reason='verified_identity_and_bytes',
                                         key=expected_device, physical_devices=found['physical_devices'],
                                         **{name: found.get(name, '') for name in ['mount', 'model', 'transport', 'filesystem', 'physical']}))
            atomic_json(receipt_path, receipt)
            decision(destination, True, 'verified_identity_and_bytes', found)
            return receipt
        except OSError as error:
            if error.errno in (errno.ENOSPC, errno.EDQUOT, errno.ENODEV, errno.ENOENT, errno.EIO, errno.EROFS):
                return exclude('insufficient_space' if error.errno in (errno.ENOSPC, errno.EDQUOT) else 'copy_device_io_failure')
            return exclude('copy_io_error')
        except ValueError:
            remove_owned_partial(partial)
            raise
        finally:
            if source_fd >= 0:
                os.close(source_fd)
            if target_fd >= 0:
                os.close(target_fd)


def copy_set(manifest_path, candidates, output, external=False, floor_percent=DEFAULT_FLOOR_PERCENT):
    owner_window()
    manifest = read_json(manifest_path)
    check_source(manifest)
    source_device = hardware(Path(manifest['source']).parent, external)
    if candidates is None:
        command = [os.environ['ANEMONE_STORAGE_ENGINE'], 'storage-scan']
        if external:
            command.append('--replica_external=1')
        scan = command_json(command)
        candidates = []
        for item in scan['candidates']:
            mount = Path(item['path'])
            parent = (Path.home() if str(mount) == '/' else mount) / 'anemone_engram_rnd/replicas'
            candidates.append(dict(path=str(parent / (manifest['pack_id'] + '.anemone'))))
    selection = select_devices(candidates, manifest['bytes'], external, floor_percent,
                               source_device.get('physical_devices', []), resume=True)
    results = [decision(manifest['source'], True, 'eligible', source_device)] if source_device.get('admitted') else []
    for selected in selection['decisions']:
        path = Path(selected['path'])
        if not selected['included']:
            if path.parent.exists() and 'anemone_engram_rnd' in path.parent.resolve().parts:
                remove_owned_partial(path.with_name(path.name + '.partial'))
            results.append(selected)
            continue
        require('anemone_engram_rnd' in path.parent.parts, 'new replica directory required')
        path.parent.mkdir(parents=True, exist_ok=True)
        receipt = path.with_name(path.name + '.replica.json')
        copied = copy_replica(manifest_path, path, receipt, selected['key'], external, floor_percent)
        results.append(copied['decision'] if copied.get('complete') else copied)
        atomic_json(output, dict(schema=SCHEMA, replica_policy=selection['replica_policy'],
                                 complete=False, decisions=results))
    final = dict(schema=SCHEMA, replica_policy=selection['replica_policy'], complete=True, decisions=results,
                 devices=[r for r in results if r['included']])
    atomic_json(output, final)
    return final


def tune(plan, output, retune=False):
    """Explicit per-action/per-device search. Workers report actual results, including refused modes."""
    owner_window()
    require(plan.get('schema') == SCHEMA and len(plan['points']) <= 4096, 'invalid search plan')
    if Path(output).exists() and not retune:
        raise ValueError('result already exists; explicit --retune required')
    require(plan['action'] in ('turbo_load', 'turbo_engram'), 'action')
    policy = plan.get('replica_policy', {})
    external = bool(policy.get('external', 0))
    floor = policy.get('floor_percent', DEFAULT_FLOOR_PERCENT)
    require(type(plan['hardware']) is list and len(plan['hardware']) <= 64, 'hardware limit')
    candidates = [dict(path=item['path'], key=item['key']) for item in plan['hardware']]
    selection = select_devices(candidates, 0, external, floor)
    admitted_keys = {item['key'] for item in selection['devices']}
    identities = {item['path']: item for item in selection['decisions']}
    planned_keys = {item['key'] for item in plan['hardware']}
    results = []
    deadline = time.monotonic() + min(1700, plan['seconds'])
    for point in plan['points']:
        owner_window()
        require(type(point.get('devices')) is list and point['devices'] and
                set(point['devices']) <= planned_keys, 'point must declare its device keys')
        if not set(point['devices']) <= admitted_keys:
            results.append(dict(point=point, status=125, reason='device_disappeared_or_unknown',
                                exact_bytes=False, supported_runtime=False))
            continue
        require(time.monotonic() < deadline, 'search deadline')
        require(point['argv'] and all(type(v) is str and '\0' not in v for v in point['argv']), 'argv')
        for path, expected in point['pins'].items():
            with open(path, 'rb') as stream:
                require(hashlib.file_digest(stream, 'sha256').hexdigest() == expected, 'worker pin changed')
        descriptor, temporary = tempfile.mkstemp(prefix='storage-point-', dir=Path(output).parent)
        child = None
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                def bounded_output():
                    resource.setrlimit(resource.RLIMIT_FSIZE, (LIMIT, LIMIT))
                child = subprocess.Popen(point['argv'], stdout=stream, stderr=subprocess.STDOUT,
                                         start_new_session=True, preexec_fn=bounded_output)
                try:
                    status = child.wait(timeout=min(point['seconds'], deadline - time.monotonic()))
                except BaseException:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=5)
                    raise
            with open(temporary, 'rb') as stream:
                data = stream.read(LIMIT + 1)
            require(len(data) <= LIMIT, 'worker output too large')
            # Failed/refused points are retained verbatim and never enter winner selection.
            result = dict(point=point, status=status, output=data.decode(errors='replace'))
            if status == 0:
                result.update(json.loads(data))
                require(result['status'] == 0, 'worker status disagreement')
            results.append(result)
            atomic_json(output, dict(schema=SCHEMA, action=plan['action'], identities=identities,
                                     plan_sha256=digest(canonical(plan)), replica_policy=selection['replica_policy'],
                                     decisions=selection['decisions'], complete=False, points=results))
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)
    # Available bytes naturally change during measurement; revalidate the stable key and policy only.
    final_selection = select_devices(candidates, 0, external, floor)
    final_keys = {item['key'] for item in final_selection['devices']}
    for result in results:
        if not set(result['point']['devices']) <= final_keys:
            result['supported_runtime'] = False
            result['reason'] = 'hardware_identity_mismatch_retune_required'
    final = dict(schema=SCHEMA, action=plan['action'], identities=identities,
                 plan_sha256=digest(canonical(plan)), replica_policy=final_selection['replica_policy'],
                 decisions=final_selection['decisions'], complete=True, points=results)
    atomic_json(output, final)
    return final


def publish(profile, load_points, engram_points, output, retune=False):
    """main() calls this only after native qualification. It never invokes measurement workers."""
    require(profile.get('schema') == SCHEMA and profile.get('enabled') == 1, 'qualified profile required')
    require(type(profile.get('decisions')) is list and len(profile['decisions']) <= 64, 'selection decisions required')
    policy = profile['replica_policy']
    require(policy.get('external') in (0, 1) and type(policy.get('floor_percent')) is int and
            0 <= policy['floor_percent'] < 100, 'replica policy')
    for item in profile['decisions']:
        require(item.get('reason') in REPLICA_REASONS and item.get('included') in (0, 1), 'invalid saved decision')
        require(bool(item['included']) == (item['reason'] in ('eligible', 'verified_identity_and_bytes')),
                'decision reason disagrees with status')
        if item['included']:
            require(any(d['path'] == item['path'] and d['key'] == item.get('key') for d in profile['devices']),
                    'included decision is absent from runtime set')
    for item in profile['devices']:
        require(any(d['path'] == item['path'] and d['included'] == 1 and d.get('key') == item['key']
                    for d in profile['decisions']), 'unrecorded device decision')
    load = select_points('turbo_load', load_points)
    engram = select_points('turbo_engram', engram_points)
    require(profile['turbo_load'] == load['winner']['runtime_config'], 'load configuration differs from winner')
    require(profile['turbo_engram'] == engram['winner']['runtime_config'], 'lookup configuration differs from winner')
    for item in profile['devices']:
        require(hardware(Path(item['path']).parent, bool(policy['external']))['key'] == item['key'], 'hardware changed')
        require(item['load_weight'] == load['winner']['weights'][item['key']] and
                item['engram_weight'] == engram['winner']['weights'][item['key']], 'weights differ from aggregate winner')
    require(digest(Path(profile['manifest']).read_bytes()) == profile['manifest_sha256'], 'manifest changed')
    if Path(output).exists() and not retune:
        require(canonical(read_json(output)) == canonical(profile), 'profile exists; explicit --retune required')
        return
    # Validate the complete candidate with the engine before publishing the serving-runtime file.
    descriptor, temporary = tempfile.mkstemp(prefix='.storage-validate-', dir=Path(output).parent)
    os.close(descriptor)
    try:
        atomic_json(temporary, profile)
        require(command_json([os.environ['ANEMONE_STORAGE_ENGINE'], 'storage-profile', temporary], timeout=1730)['valid'],
                'engine dry-run refused candidate')
        atomic_json(output, profile)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def select_points(action, points):
    """Compare measured aggregate objectives. Solo rates are never converted into assumed shares."""
    require(action in ('turbo_load', 'turbo_engram'), 'action')
    eligible = [p for p in points if p.get('status') == 0 and p.get('exact_bytes') is True
                and p.get('aggregate') is True and p.get('native_qualified') is True
                and p.get('supported_runtime') is True]
    require(eligible, 'no qualified aggregate point')
    devices = set(eligible[0]['devices'])
    require(devices and all(set(p['devices']) == devices for p in eligible), 'aggregate device sets differ')
    for device in devices:
        require(any(p.get('status') == 0 and p.get('devices') == [device]
                    and p.get('exact_bytes') is True for p in points), 'missing solo measurement')
    if action == 'turbo_load':
        require(all(p.get('page_locked') is True for p in eligible), 'page-locked measurement required')
        objective = lambda p: (p['startup_to_serving_seconds'], -p['gbps'])
    else:
        require(all(p.get('decode_slowdown', 1) <= 0 and sorted(p['step_p99_ms']) == ['1', '2', '4', '8']
                    for p in eligible), 'native decode/C1-C8 proof required')
        objective = lambda p: (max(p['step_p99_ms'].values()), max(p['step_p50_ms'].values()))
    ranked = sorted(eligible, key=objective)
    return dict(winner=ranked[0], runner_up=ranked[1] if len(ranked) > 1 else None,
                objective='startup_then_gbps' if action == 'turbo_load' else 'worst_C1_C8_p99_then_p50',
                measured_points=points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    choose = commands.add_parser('select')
    choose.add_argument('action', choices=['turbo_load', 'turbo_engram'])
    choose.add_argument('points'); choose.add_argument('output')
    install = commands.add_parser('publish')
    install.add_argument('profile'); install.add_argument('load_points'); install.add_argument('engram_points')
    install.add_argument('output'); install.add_argument('--retune', action='store_true')
    arguments = parser.parse_args()
    if arguments.command == 'select':
        atomic_json(arguments.output, select_points(arguments.action, read_json(arguments.points)))
    else:
        publish(read_json(arguments.profile), read_json(arguments.load_points), read_json(arguments.engram_points),
                arguments.output, arguments.retune)


if __name__ == '__main__':
    main()
