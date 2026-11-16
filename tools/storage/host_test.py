#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Linux CPU functional tests of real engine validation and crash-safe persistence."""
import copy
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import time
import unittest
from unittest import mock
import io
import contextlib

import manager

ENGINE = os.environ['ANEMONE_STORAGE_ENGINE']


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='anemone-storage-test-')
        self.root = Path(self.temporary.name)
        self.data = b'engine-exact-storage-row-test' * 512
        header = struct.pack('<8sII64sQQQIIIIQ', b'ANEMONE\0', 2, 4096, b'fixture',
                             len(self.data), 4096, 0, 1, 1, 0, 0, 1)
        self.data = header + self.data[len(header):]
        self.source = self.root / 'source.anemone'
        self.source.write_bytes(self.data)
        self.replica = self.root / 'replica.anemone'
        self.replica.write_bytes(self.data)
        self.manifest_path = self.root / 'manifest.json'
        self.manifest = dict(schema=1, source=str(self.source), pack_id='0000000000000001', bytes=len(self.data),
                             chunk_bytes=4096, chunks=[manager.digest(self.data[i:i + 4096])
                                                      for i in range(0, len(self.data), 4096)])
        manager.atomic_json(self.manifest_path, self.manifest)
        self.hardware = manager.hardware(self.root)
        self.key = self.hardware['key']
        self.profile = dict(schema=manager.SCHEMA, replica_policy=dict(external=0, floor_percent=20), enabled=1, created_unix=int(time.time()), source=str(self.source),
                            manifest=str(self.manifest_path), manifest_sha256=manager.digest(self.manifest_path.read_bytes()),
                            devices=[dict(path=str(self.replica), key=self.key, load_weight=3, engram_weight=7)],
                            turbo_load=dict(enabled=1, threads=1, mechanism='direct_pread'),
                            turbo_engram=dict(enabled=1, ram_percent=50, cache_mib=1, io=0, mechanism='direct_pread'))

    def tearDown(self):
        self.temporary.cleanup()

    def profile_document(self, profile=None):
        value = copy.deepcopy(self.profile if profile is None else profile)
        if 'decisions' not in value:
            value['decisions'] = []
            for item in value['devices']:
                parent = Path(item['path']).parent
                found = manager.hardware(parent)
                value['decisions'].append(dict(path=item['path'], key=item['key'], included=1,
                    reason='verified_identity_and_bytes',
                    **{name: found.get(name, '') for name in ['mount', 'model', 'transport', 'filesystem', 'physical']},
                    physical_devices=found.get('physical_devices', [])))
        return value

    def check(self, profile=None, reason='valid', status=0):
        path = self.root / 'profile.json'
        manager.atomic_json(path, self.profile_document(profile))
        result = subprocess.run([ENGINE, 'storage-profile', str(path)], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, status, result.stderr + result.stdout)
        parsed = json.loads(result.stdout)
        self.assertEqual(parsed['reason'], reason)
        self.stderr = result.stderr
        return parsed

    def test_missing_profile_startup_hint_is_visible_and_inert(self):
        path = self.root / 'absent-profile.json'
        result = subprocess.run([ENGINE, 'load', '--storage_profile=' + str(path), str(self.source)],
                                capture_output=True, text=True, timeout=20)
        expected = 'storage: no tuned storage profile: run `anemone autotune` before serving for best load/lookup performance\n'
        self.assertEqual(result.stderr.count(expected), 1)
        self.assertNotEqual(result.returncode, 0)  # Tiny invalid pack stops before any backend/card opens.

    def test_real_reload_and_no_benchmark(self):
        # Poison executable search. The engine validator cannot spawn a benchmark or helper.
        sentinel = self.root / 'benchmark-was-started'
        script = self.root / 'engram_replay'
        script.write_text('#!/bin/sh\n/usr/bin/touch ' + str(sentinel) + '\nexit 1\n')
        script.chmod(0o700)
        previous = os.environ.get('PATH', '')
        os.environ['PATH'] = str(self.root)
        try:
            self.check()
            self.check()
            self.assertFalse(sentinel.exists())
        finally:
            os.environ['PATH'] = previous

    def test_startup_reload_uses_saved_profile_without_helpers(self):
        # The ordinary load entry must reload too, not only the model-free validator.
        # This deliberately invalid fixture pack stops before any backend/card opens.
        config = self.root / 'anemone'
        config.mkdir()
        path = config / 'storage-profile.json'
        manager.atomic_json(path, self.profile_document())
        sentinel = self.root / 'startup-helper-was-started'
        for name in ['engram_replay', 'fio', 'dd', 'python3', 'anemone-storage-tune']:
            helper = self.root / name
            helper.write_text('#!/bin/sh\n/usr/bin/touch ' + str(sentinel) + '\nexit 1\n')
            helper.chmod(0o700)
        environment = dict(os.environ, XDG_CONFIG_HOME=str(self.root), PATH=str(self.root))
        environment.pop('ANEMONE_STORAGE_PROFILE', None)

        def startup(extra=()):
            result = subprocess.run([ENGINE, 'load', *extra, str(self.source)], env=environment,
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
            self.assertTrue('the step "open the pack" failed:' in result.stderr or
                            'the step "open the copies" failed:' in result.stderr,
                            result.stderr + result.stdout)
            self.assertFalse(sentinel.exists(), result.stderr + result.stdout)
            return result.stderr

        identity = manager.digest(path.read_bytes())
        for _ in range(2):
            text = startup()
            self.assertIn('storage profile: ' + identity + ' age_seconds=', text)
            self.assertIn('load=1 engram=0 replicas=1 excluded=0 valid', text)
        explicit_off = startup(['--storage_profile=off'])
        self.assertNotIn('storage profile:', explicit_off)
        self.assertNotIn('autotune', explicit_off)
        path.unlink()
        missing = startup()
        self.assertIn('storage profile: fallback missing', missing)
        self.assertIn('storage: no tuned storage profile: run `anemone autotune`', missing)
        bad = copy.deepcopy(self.profile)
        bad['schema'] = 999
        manager.atomic_json(path, self.profile_document(bad))
        self.assertIn('storage profile: fallback schema_mismatch', startup())
        bad = copy.deepcopy(self.profile)
        bad['devices'][0]['key'] = '0' * 64
        manager.atomic_json(path, self.profile_document(bad))
        self.assertIn('storage profile: fallback no_verified_replicas', startup())

    def test_profile_disk_tables_admit_step_io_after_it_applies(self):
        # Review findings 3 and 11, and the order of run_program: the profile's turbo_engram puts the tables on disk
        # before step I/O's guard runs, so serving with --engram_step_io=1 (no profile argument) is admitted and
        # startup goes on to the tables' file; with the profile off, the same command is refused before any file.
        path = self.root / 'profile.json'
        manager.atomic_json(path, self.profile_document())
        tables = self.root / 'absent-tables.json'

        def serve(profile):
            return subprocess.run([ENGINE, 'serve', '--storage_profile=' + profile, '--engram_step_io=1',
                                   str(tables), str(self.source)], capture_output=True, text=True, timeout=20)
        refusal = '--engram_step_io=1 and --engram_hot_set need disk tables'
        admitted = serve(str(path))
        self.assertIn('load=1 engram=1 replicas=1 excluded=0 valid', admitted.stderr)
        self.assertNotIn(refusal, admitted.stderr)
        self.assertIn('the step "open the tables and the socket" failed:', admitted.stderr)
        self.assertEqual(admitted.returncode, 1, admitted.stderr)
        refused = serve('off')
        self.assertIn(refusal, refused.stderr)
        self.assertNotIn('open the tables', refused.stderr)
        self.assertNotEqual(refused.returncode, 0, refused.stderr)

    def exl3_fixture(self):
        # An EXL3 map whose one shard holds a fixture pack of whole map blocks, the source manifest of its logical bytes,
        # and a profile whose source, and only device, is the map.
        block = 4096
        data = (b'engine-exact-storage-exl3-row-' * block)[:4 * block]
        header = struct.pack('<8sII64sQQQIIIIQ', b'ANEMONE\0', 2, block, b'fixture', len(data), block, 0, 1, 1, 0, 0, 2)
        self.exl3_data = header + data[len(header):]
        self.shard = self.root / 'shard.bin'
        self.shard.write_bytes(self.exl3_data)
        sources = [b'', str(self.shard).encode()]
        map_path = self.root / 'model.exl3map'
        map_path.write_bytes(struct.pack('<8sQII40s', b'ANEXL3\0\0', len(self.exl3_data), len(sources), 1, bytes(40)) +
                             b''.join(source.ljust(512, b'\0') for source in sources) +
                             struct.pack('<QQQII', 0, len(self.exl3_data), 0, 1, 0))
        manifest_path = self.root / 'exl3-manifest.json'
        manager.atomic_json(manifest_path, dict(schema=1, source=str(map_path), pack_id='0000000000000002',
                                                bytes=len(self.exl3_data), chunk_bytes=block,
                                                chunks=[manager.digest(self.exl3_data[i:i + block])
                                                        for i in range(0, len(self.exl3_data), block)]))
        profile = copy.deepcopy(self.profile)
        profile.update(source=str(map_path), manifest=str(manifest_path),
                       manifest_sha256=manager.digest(manifest_path.read_bytes()),
                       devices=[dict(path=str(map_path), key=self.key, load_weight=3, engram_weight=7)])
        return map_path, profile

    def test_exl3_map_source_is_verified_by_its_logical_bytes(self):
        # An EXL3 profile's source is its map. Its identity and every manifest chunk are the logical pack's, read
        # through the map reader (before: the map file's own size and first bytes, so such a profile never validated).
        _, profile = self.exl3_fixture()
        self.assertEqual(self.check(profile)['reason'], 'valid')
        changed = bytearray(self.exl3_data)
        changed[-1] ^= 1
        self.shard.write_bytes(changed)
        self.check(profile, 'no_verified_replicas', 1)

    def test_exl3_map_is_no_disk_lookup_source(self):
        # The disk store reads packs, not an EXL3 map: a profile whose only verified device is the map loads from it
        # and keeps lookup off, with its reason (before: the map was given as an --engram_sources path).
        map_path, profile = self.exl3_fixture()
        path = self.root / 'exl3-profile.json'
        manager.atomic_json(path, self.profile_document(profile))
        result = subprocess.run([ENGINE, 'serve', '--storage_profile=' + str(path), str(self.root / 'absent-tables.json'),
                                 str(map_path)], capture_output=True, text=True, timeout=20)
        self.assertIn('storage profile: fallback engram_sources_none', result.stderr)
        self.assertIn('load=1 engram=0 replicas=1 excluded=0 valid', result.stderr)
        self.assertNotEqual(result.returncode, 0, result.stderr)

    def test_fallbacks(self):
        cases = [(['schema'], 999, 'schema_mismatch'), (['enabled'], 0, 'disabled_or_unqualified'),
                 (['created_unix'], int(time.time()) + 86400, 'invalid_profile_fields'),
                 (['manifest_sha256'], '0' * 64, 'manifest_missing_or_digest_mismatch'),
                 (['turbo_load', 'threads'], 0, 'unsupported_action_parameters'),
                 (['turbo_load', 'mechanism'], 'unmeasured', 'unsupported_action_parameters'),
                 (['turbo_engram', 'mechanism'], 'mmap', 'unsupported_action_parameters'),
                 (['turbo_engram', 'ram_percent'], 100, 'unsupported_action_parameters'),
                 (['turbo_engram', 'cache_mib'], 4097, 'unsupported_action_parameters'),
                 (['turbo_engram', 'io'], 9, 'unsupported_action_parameters'),
                 (['devices'], [], 'no_verified_replicas')]
        for keys, value, reason in cases:
            with self.subTest(keys=keys, value=value):
                candidate = copy.deepcopy(self.profile)
                target = candidate
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                self.check(candidate, reason, 1)

    def test_missing_unreadable_schema_and_duplicates(self):
        path = self.root / 'bad.json'
        result = subprocess.run([ENGINE, 'storage-profile', str(path)], capture_output=True, text=True, timeout=20)
        self.assertEqual(json.loads(result.stdout)['reason'], 'missing')
        for data in [b'{', b'{"schema":1,"schema":1}', b'[]', b'null', b' ' * (manager.LIMIT + 1)]:
            path.write_bytes(data)
            result = subprocess.run([ENGINE, 'storage-profile', str(path)], capture_output=True, text=True, timeout=20)
            self.assertEqual(json.loads(result.stdout)['reason'], 'unreadable_or_invalid_json')
        manager.atomic_json(path, self.profile_document())
        path.chmod(0o666)
        result = subprocess.run([ENGINE, 'storage-profile', str(path)], capture_output=True, text=True, timeout=20)
        self.assertEqual(json.loads(result.stdout)['reason'], 'unreadable_or_invalid_json')

    def test_identity_mismatch_and_duplicate_physical(self):
        self.profile['devices'][0]['key'] = '0' * 64
        self.check(reason='no_verified_replicas', status=1)
        self.profile['devices'][0]['key'] = self.key
        self.profile['devices'].append(copy.deepcopy(self.profile['devices'][0]))
        parsed = self.check(reason='degraded_replicas_excluded', status=0)
        self.assertEqual((parsed['replicas'], parsed['excluded']), (1, 1))
        duplicate = 'storage: replica %s excluded: duplicate_physical_device (shares physical device '
        self.assertIn(duplicate % self.replica, self.stderr)
        self.assertIn(' with kept replica %s)\n' % self.replica, self.stderr)
        # Two paths to one disk: the first verified path stays, the other is named with the shared identity.
        second = self.root / 'second-path.anemone'
        second.write_bytes(self.data)
        self.profile['devices'][1]['path'] = str(second)
        parsed = self.check(reason='degraded_replicas_excluded', status=0)
        self.assertEqual((parsed['replicas'], parsed['excluded']), (1, 1))
        self.assertIn(duplicate % second, self.stderr)
        self.assertIn(' with kept replica %s)\n' % self.replica, self.stderr)
        # A different decision for the same path contradicts the manifest.
        document = self.profile_document()
        document['devices'][1]['path'] = str(self.replica)
        document['decisions'][1] = dict(document['decisions'][0], reason='eligible')
        self.check(document, reason='invalid_decision_manifest', status=1)

    def test_partial_replica_not_admitted(self):
        partial = self.replica.with_name('replica.partial')
        partial.write_bytes(self.data)
        self.profile['devices'][0]['path'] = str(partial)
        self.check(reason='invalid_device_parameters', status=1)

    def test_new_source_never_reuses_old_model_replicas(self):
        data = bytearray(self.data)
        struct.pack_into('<Q', data, 120, 2)
        self.source.write_bytes(data)
        self.check(reason='source_identity_mismatch', status=1)

    def test_short_read_removed_corruption_manifest(self):
        self.replica.write_bytes(self.data[:-1])
        self.check(reason='no_verified_replicas', status=1)
        self.replica.write_bytes(b'x' + self.data[1:])
        self.check(reason='no_verified_replicas', status=1)
        self.replica.unlink()
        self.check(reason='no_verified_replicas', status=1)
        self.manifest_path.unlink()
        self.check(reason='manifest_missing_or_digest_mismatch', status=1)

    def test_stale_replica_degrades_to_verified_device(self):
        path = os.environ.get('ANEMONE_STORAGE_STALE_FIXTURE')
        self.assertIsNotNone(path, 'point to an existing pack on a second physical device')
        second = manager.hardware(Path(path).parent)
        self.assertNotEqual(second['key'], self.key)
        self.profile['devices'].append(dict(path=path, key=second['key'], load_weight=1, engram_weight=2))
        result = self.check(reason='degraded_replicas_excluded')
        self.assertEqual((result['replicas'], result['excluded']), (1, 1))

    def test_atomic_crash_boundaries(self):
        path = self.root / 'persistent.json'
        for boundary, expected in [('file_fsynced', 'old'), ('renamed', 'new'), ('directory_fsynced', 'new')]:
            manager.atomic_json(path, dict(value='old'))
            child = os.fork()
            if child == 0:
                manager.atomic_json(path, dict(value='new'),
                                    lambda step: os._exit(71) if step == boundary else None)
                os._exit(72)
            _, status = os.waitpid(child, 0)
            self.assertEqual(os.waitstatus_to_exitcode(status), 71)
            self.assertEqual(manager.read_json(path), dict(value=expected))
        # An ordinary exception before rename preserves the old profile too.
        manager.atomic_json(path, dict(value='old'))
        with self.assertRaises(RuntimeError):
            manager.atomic_json(path, dict(value='new'), lambda _: (_ for _ in ()).throw(RuntimeError('crash')))
        self.assertEqual(manager.read_json(path), dict(value='old'))

    def test_copy_resume_and_idempotence(self):
        # Tiny owned fixture bytes, never a learned table or system cache eviction.
        directory = self.root / 'anemone_engram_rnd'
        directory.mkdir()
        target = directory / 'copy.anemone'
        partial = target.with_name(target.name + '.partial')
        partial.write_bytes(self.data[:4096] + bytes(len(self.data) - 4096))
        receipt = directory / 'receipt.json'
        manager.copy_replica(self.manifest_path, target, receipt, self.key)
        self.assertEqual(target.read_bytes(), self.data)
        manager.copy_replica(self.manifest_path, target, receipt, self.key)
        self.assertTrue(manager.read_json(receipt)['complete'])
        target.chmod(0o400)
        manager.copy_replica(self.manifest_path, target, receipt, self.key)
        self.assertEqual(target.read_bytes(), self.data)
        target.chmod(0o600)
        target.write_bytes(bytes(len(self.data)))
        result = manager.copy_replica(self.manifest_path, target, receipt, self.key)
        self.assertFalse(result['complete'])
        self.assertEqual(result['reason'], 'finalized_replica_unverified')
        self.assertEqual(target.read_bytes(), bytes(len(self.data)))

    def test_aggregate_selection_never_invents_weights(self):
        points = [dict(status=0, devices=[key], exact_bytes=True) for key in ['a', 'b']]
        base = dict(status=0, devices=['a', 'b'], exact_bytes=True, aggregate=True,
                    native_qualified=True, supported_runtime=True, page_locked=True,
                    startup_to_serving_seconds=10, gbps=6, weights=dict(a=3, b=1))
        points += [base, dict(base, startup_to_serving_seconds=11, gbps=7, weights=dict(a=1, b=1))]
        self.assertEqual(manager.select_points('turbo_load', points)['winner']['weights'], dict(a=3, b=1))
        with self.assertRaisesRegex(ValueError, 'solo'):
            manager.select_points('turbo_load', points[1:])
        with self.assertRaisesRegex(ValueError, 'qualified'):
            manager.select_points('turbo_load', [dict(base, native_qualified=False)])

    def test_copy_lock_and_receipt_cannot_escape_directory(self):
        directory = self.root / 'anemone_engram_rnd'
        directory.mkdir()
        target = directory / 'copy.anemone'
        sentinel = self.root / 'sentinel'
        sentinel.write_bytes(b'unchanged')
        lock = directory / ('.' + target.name + '.copy.lock')
        lock.symlink_to(sentinel)
        with self.assertRaises(OSError):
            manager.copy_replica(self.manifest_path, target, directory / 'receipt.json', self.key)
        self.assertEqual(sentinel.read_bytes(), b'unchanged')
        lock.unlink()
        with self.assertRaisesRegex(ValueError, 'receipt'):
            manager.copy_replica(self.manifest_path, target, sentinel, self.key)
        self.assertEqual(sentinel.read_bytes(), b'unchanged')

    def test_replica_publication_never_overwrites(self):
        partial = self.root / 'partial'
        final = self.root / 'final'
        partial.write_bytes(b'new')
        final.write_bytes(b'old')
        with self.assertRaises(FileExistsError):
            manager.rename_no_replace(partial, final)
        self.assertEqual(final.read_bytes(), b'old')
        self.assertEqual(partial.read_bytes(), b'new')
        final.unlink()
        manager.rename_no_replace(partial, final)
        self.assertEqual(final.read_bytes(), b'new')
        self.assertFalse(partial.exists())

    def test_insufficient_space_excludes_without_partial(self):
        directory = self.root / 'anemone_engram_rnd'
        directory.mkdir()
        target = directory / 'copy.anemone'
        partial = target.with_name(target.name + '.partial')
        partial.write_bytes(self.data[:4096])
        refused = dict(self.hardware, admitted=False, reason='insufficient_space')
        with mock.patch.object(manager, 'copy_room', return_value=refused):
            result = manager.copy_replica(self.manifest_path, target, directory / 'receipt.json', self.key)
        self.assertFalse(result['complete'])
        self.assertEqual(result['reason'], 'insufficient_space')
        self.assertFalse(partial.exists())
        self.assertFalse(target.exists())

    def test_space_rechecked_before_publication(self):
        directory = self.root / 'anemone_engram_rnd'
        directory.mkdir()
        target = directory / 'copy.anemone'
        accepted = dict(self.hardware, admitted=True, reason='eligible')
        refused = dict(self.hardware, admitted=False, reason='insufficient_space')
        with mock.patch.object(manager, 'copy_room', side_effect=[accepted, refused]) as probe:
            result = manager.copy_replica(self.manifest_path, target, directory / 'receipt.json', self.key)
        self.assertEqual(probe.call_count, 2)
        self.assertFalse(result['complete'])
        self.assertFalse(target.exists())
        self.assertFalse(target.with_name(target.name + '.partial').exists())

    def test_remaining_device_set_after_policy_exclusions(self):
        # Controlled policy facts: actual sysfs classification is separately tested in local_device_test.
        base = dict(self.hardware, admitted=True, reason='eligible')
        candidates = [dict(path=str(self.root / name / 'copy.anemone'))
                      for name in ['internal', 'usb', 'network', 'alias', 'small']]
        values = [base, dict(base, admitted=False, reason='external_device_requires_override'),
                  dict(base, admitted=False, reason='network_filesystem_or_transport'), base,
                  dict(base, admitted=False, reason='insufficient_space')]
        with mock.patch.object(manager, 'copy_room', side_effect=values):
            selected = manager.select_devices(candidates, len(self.data))
        self.assertEqual(len(selected['devices']), 1)
        self.assertEqual([d['reason'] for d in selected['decisions']],
                         ['eligible', 'external_device_requires_override', 'network_filesystem_or_transport',
                          'duplicate_physical_device', 'insufficient_space'])
        usb = dict(base, key='1' * 64, physical='0000000000000002', physical_devices=['0000000000000002'])
        with mock.patch.object(manager, 'copy_room', side_effect=[base, usb, values[2], base, values[4]]) as probe:
            forced = manager.select_devices(candidates, len(self.data), external=True)
        self.assertEqual(len(forced['devices']), 2)
        self.assertEqual(forced['decisions'][2]['reason'], 'network_filesystem_or_transport')
        self.assertTrue(all(call.args[2] is True for call in probe.call_args_list))

    def test_disappearing_device_degrades_and_remains_reported(self):
        missing = str(self.root / 'removed-mount' / 'replica.anemone')
        self.profile['devices'].append(dict(path=missing, key='1' * 64, load_weight=1, engram_weight=1))
        candidate = self.profile_document()
        candidate['decisions'][-1].update(mount='/mnt/removed', model='fixture_ssd', transport='nvme',
                                        physical_devices=['0000000000000002'])
        result = self.check(candidate, reason='degraded_replicas_excluded')
        self.assertEqual((result['replicas'], result['excluded']), (1, 1))
        path = self.root / 'profile.json'
        report = subprocess.run([ENGINE, 'storage-report', str(path), '--metrics=1'],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(report.returncode, 0, report.stderr)
        self.assertIn('storage: replica /mnt/removed (fixture_ssd, nvme, 0000000000000002) excluded: device_disappeared_or_unknown' + chr(10), report.stderr)
        self.assertIn('anemone_storage_replica_info{mount="/mnt/removed",role="load",status="excluded",reason="device_disappeared_or_unknown"} 1' + chr(10), report.stdout)
        self.assertLessEqual(report.stdout.count('anemone_storage_replica_info{'), 128)

    def test_excluded_network_decision_is_never_described_again(self):
        # An enabled profile's excluded network decision keeps its recorded reason: its path (an idle automount on a
        # real host) is never stat'ed or statfs'ed at startup.
        candidate = self.profile_document()
        candidate['decisions'].append(dict(path='/nonexistent-nas/anemone_engram_rnd/replicas/copy.anemone',
                                           key='2' * 64, included=0, reason='network_filesystem_or_transport',
                                           mount='/nonexistent-nas', model='fixture_nas', transport='network',
                                           filesystem='nfs4', physical='', physical_devices=[]))
        result = self.check(candidate, reason='degraded_replicas_excluded')
        self.assertEqual((result['replicas'], result['excluded']), (1, 1))
        self.assertIn('excluded: network_filesystem_or_transport filesystem=nfs4' + chr(10), self.stderr)
        self.assertNotIn('replica decision changed', self.stderr)
        self.assertNotIn('device_disappeared_or_unknown', self.stderr)

    def test_excluded_decisions_that_read_alike_keep_the_profile(self):
        # Two refused mounts whose paths are no UTF-8 are both written with U+FFFD and read alike; two excluded
        # decisions for one path contradict nothing, so only they are excluded.
        candidate = self.profile_document()
        for model, physical in (('fixture_usb_a', '0000000000000002'), ('fixture_usb_b', '0000000000000003')):
            candidate['decisions'].append(dict(path='/media/caf\ufffd/anemone_engram_rnd/replicas/copy.anemone',
                                               key='3' * 64, included=0, reason='destination_directory_policy',
                                               mount='/media/caf\ufffd', model=model, transport='usb',
                                               filesystem='ext4', physical=physical, physical_devices=[physical]))
        result = self.check(candidate, reason='degraded_replicas_excluded')
        self.assertEqual((result['replicas'], result['excluded']), (1, 2))

    def test_autotune_refuses_a_path_that_is_no_utf8_before_any_copy(self):
        source = os.fsencode(str(self.root)) + b'/caf\xe9.anemone'
        with open(source, 'wb') as stream:
            stream.write(self.data)
        profile = self.root / 'refused-profile.json'
        result = subprocess.run([os.fsencode(ENGINE), b'autotune', source,
                                 b'--tokens=' + os.fsencode(str(self.root / 'tokens.u32')),
                                 b'--profile=' + os.fsencode(str(profile))], capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn(b'autotune: the source path is no UTF-8, so the profile cannot record it', result.stderr)
        self.assertFalse(profile.exists())

    def test_autotune_checks_the_real_source_path_for_utf8(self):
        # The manifest and profile record realpath(source): a UTF-8 link to a real path that is no UTF-8 is refused
        # before any copy; a link whose name is no UTF-8 to a UTF-8 file passes that check (and stops later, at this
        # fixture's pack).
        real = os.fsencode(str(self.root)) + b'/caf\xe8.anemone'
        with open(real, 'wb') as stream:
            stream.write(self.data)
        root = os.fsencode(str(self.root))
        for link, refused in ((root + b'/linked.anemone', True), (root + b'/link\xe9.anemone', False)):
            os.symlink(real if refused else os.fsencode(str(self.source)), link)
            profile = self.root / ('linked-profile.json' if refused else 'named-profile.json')
            result = subprocess.run([os.fsencode(ENGINE), b'autotune', link,
                                     b'--tokens=' + os.fsencode(str(self.root / 'tokens.u32')),
                                     b'--profile=' + os.fsencode(str(profile))], capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(b'autotune: the source path is no UTF-8, so the profile cannot record it' in result.stderr,
                             refused)
            self.assertFalse(profile.exists())

    def test_original_source_loss_uses_verified_replica(self):
        candidate = self.profile_document()
        self.source.unlink()
        self.check(candidate)

    def test_report_keeps_exclusions_when_every_device_is_lost(self):
        candidate = self.profile_document()
        self.replica.unlink()
        self.check(candidate, reason='no_verified_replicas', status=1)
        report = subprocess.run([ENGINE, 'storage-report', str(self.root / 'profile.json'), '--metrics=1'],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(report.returncode, 1)
        self.assertIn('reason="replica_missing_short_or_unverified"', report.stdout)
        self.assertNotIn('status="included"', report.stdout)

    def test_exact_product_lines_use_fixed_reasons(self):
        device = dict(mount='/mnt/test', model='fixture_ssd', transport='usb', filesystem='nfs4',
                      physical='0000000000000001', available_bytes=310)
        prefix = 'storage: replica /mnt/test (fixture_ssd, usb, 0000000000000001) '
        cases = {
            'insufficient_space': 'excluded: insufficient_space needs_bytes=517 floor_percent=20 available_bytes=310',
            'external_device_requires_override': 'excluded: external_device_requires_override (use --replica_external=1)',
            'network_filesystem_or_transport': 'excluded: network_filesystem_or_transport filesystem=nfs4',
            'rotational_device': 'excluded: rotational_device',
            'duplicate_physical_device': 'excluded: duplicate_physical_device',
            'device_disappeared_or_unknown': 'excluded: device_disappeared_or_unknown',
            'hardware_identity_mismatch_retune_required': 'excluded: hardware_identity_mismatch_retune_required',
            'verified_identity_and_bytes': 'included: verified_identity_and_bytes load_weight=3 lookup_weight=7'}
        for reason, expected in cases.items():
            self.assertEqual(manager.product_line(device, reason, 517, 20, 3, 7), prefix + expected + chr(10))
        self.assertEqual(manager.product_line(device, 'untrusted_environment_text'), prefix + 'excluded: invalid_decision' + chr(10))

    def test_unsupported_allocation_excludes_and_cleans_partial(self):
        import errno
        directory = self.root / 'anemone_engram_rnd'
        directory.mkdir()
        target = directory / 'unsupported.anemone'
        with mock.patch.object(os, 'posix_fallocate', side_effect=OSError(errno.EOPNOTSUPP, 'unsupported')):
            result = manager.copy_replica(self.manifest_path, target, directory / 'receipt.json', self.key)
        self.assertFalse(result['complete'])
        self.assertEqual(result['reason'], 'copy_io_error')
        self.assertFalse(target.exists())
        self.assertFalse(target.with_name(target.name + '.partial').exists())

    def test_network_refused_before_python_path_access(self):
        refused = dict(self.hardware, admitted=False, reason='network_filesystem_or_transport', filesystem='nfs4')
        with mock.patch.object(manager, 'hardware', return_value=refused), \
             mock.patch.object(Path, 'exists', side_effect=AssertionError('remote stat attempted')):
            selected = manager.select_devices([dict(path='/remote/anemone_engram_rnd/copy')], len(self.data), external=True)
        self.assertEqual(selected['devices'], [])
        with mock.patch.object(manager, 'hardware', return_value=refused), \
             mock.patch.object(Path, 'resolve', side_effect=AssertionError('remote resolve attempted')):
            copied = manager.copy_replica(self.manifest_path, Path('/remote/anemone_engram_rnd/copy'),
                                          Path('/remote/anemone_engram_rnd/receipt'), self.key, external=True)
        self.assertFalse(copied['complete'])
        self.assertEqual(copied['reason'], 'network_filesystem_or_transport')

    def test_tune_excludes_changed_device_and_keeps_remaining(self):
        accepted = dict(self.hardware, admitted=True, reason='eligible')
        changed = dict(accepted, key='a' * 64, physical='0000000000000002',
                       physical_devices=['0000000000000002'])
        plan = dict(schema=manager.SCHEMA, action='turbo_load', seconds=1,
                    hardware=[dict(path='/fixture/a', key=self.key), dict(path='/fixture/b', key='b' * 64)],
                    points=[dict(devices=['b' * 64])])
        with mock.patch.object(manager, 'copy_room', side_effect=[accepted, changed, accepted, changed]):
            result = manager.tune(plan, self.root / 'tune.json')
        self.assertTrue(result['complete'])
        self.assertEqual([d['included'] for d in result['decisions']], [1, 0])
        self.assertEqual(result['decisions'][1]['reason'], 'hardware_identity_mismatch_retune_required')
        self.assertEqual(result['points'][0]['status'], 125)

    def test_model_free_saved_and_live_report(self):
        path = self.root / 'report-profile.json'
        manager.atomic_json(path, self.profile_document())
        result = subprocess.run([ENGINE, 'storage-report', str(path)], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('storage: saved replica decisions (before live revalidation)', result.stderr)
        self.assertIn('storage: live profile revalidation', result.stderr)
        self.assertIn('storage: live device scan', result.stderr)
        self.assertTrue(json.loads(result.stdout)['valid'])
        self.assertNotIn('benchmark', result.stdout)

    def test_unenumerated_saved_reason_is_rejected(self):
        candidate = self.profile_document()
        candidate['decisions'][0]['reason'] = 'untrusted_environment_text'
        self.check(candidate, reason='invalid_decision_manifest', status=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
