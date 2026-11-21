# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every shift of one in the engine's sources (.c, .h, .cu and the .cuh kernel headers), classified by what its amount
indexes, and every bit read of a right shift by a session-slot amount.

A server takes up to SERVING_SLOTS_MOST session slots, far above a 32- or 64-bit word, so no mask may be indexed by
session slot except the BATCH round tables, whose engine takes at most STEP_BATCH_MAXIMUM_SESSIONS slots. Each shift
amount is classified below; a shift whose amount is not classified, a session-slot shift outside the BATCH sites, or a
bit read `(mask >> slot) & 1` by a session-slot amount anywhere (none exists) fails. Run from
the repository root. This lexical check reads the amount's text, not its type.
"""
from collections import Counter
from pathlib import Path
import re
import sys

LITERALS = re.compile(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', re.S)
SHIFT = re.compile(r'(?<![\w.])(?:1|1u|1U|1ul|1UL|1ull|1ULL|UINT32_C\(1\)|UINT64_C\(1\)|'
                   r'\((?:uint32_t|uint64_t|size_t)\)\s*1u?)\s*<<\s*\(?\s*([\w.\->\[\]]+)(\s*%\s*\w+_BITS\b)?')
# a bit read through a right shift: (mask >> amount) & 1
BIT_READ = re.compile(r'>>\s*\(?\s*([\w.\->\[\]]+)\s*\)?\s*\)?\s*&\s*(?:1|1u|1U|1ul|1UL|1ull|1ULL)\b')
# The amounts that index something other than a session slot, by kind.
KINDS = {
    'card': ('card', 'card_count', 'segment->card', 'hand_off->to_card', 'start->card', 'pause->card', 'first->card',
             'first->card_count', 'first->segments[member].card', 'PREFIX_CARDS'),
    'rank or collective member': ('rank', 'ranks', 'members', 'source', 'arguments->rank', 'arguments->rank_count',
                                  'row->rank_count', 'stage->rank_count'),
    'storage copy': ('copy', 'copies', 'copies[miss]', 'context->copies', 'read->copy'),
    # a replica layout of one mount (replicas.c's stale receipts, a bit each): below REPLICA_LAYOUTS (3)
    'replica layout': ('layout',),
    # a /proc/vmstat counter the load stage reads (storage_autotune_load.c's vmstat_read, a bit each): below
    # AUTOTUNE_LOAD_VMSTAT_COUNTERS (5)
    'load stage page counter': ('counter',),
    'model layer or unit': ('layer', 'units[unit].layer_index', 'DS41F_DECODER_FIRST_LAYER'),
    'Engram table': ('table', 'step->blocks[index].table'),
    'row of one step or block': ('row',),
    'GPU lane or key of one warp': ('lane', 'key', 'held'),
    'bit count, word bit or table size': ('count', 'bits', 'bin', 'depth', 'index', 'chunk', 'remaining', 'slot_bits',
                                          'bucket_bits', 'draft_lookup_bucket_bits', 'decoder->available',
                                          'experts[slot]', 'first_kernel_of', 'kernel_place_of', 'wide_form',
                                          'HEAD_MMA_WIDE_THIN_FORM', 'BITS_PER_WORD', 'JPEG_WORD_BITS',
                                          'SAMPLER_WORD_BITS',
                                          'RESIZE_COEFFICIENT_BITS', 'HALF_BYTE_BITS', 'E2M1_SIGN_SHIFT',
                                          'E2M1_BFLOAT16_BIT',
                                          # the .cuh kernel headers' number formats (kernel_class_80.cuh,
                                          # kernel_mma_common.cuh): a field's or a fraction's bits
                                          'fraction_bits', 'magnitude_bit', 'is_bfloat16',
                                          'CLASS_80_BFLOAT16_FIELD_SHIFT',
                                          # a lane's not-finite marks: a bit for each output half's row of A, then
                                          # the column's (CLASS_80_COLUMN_MARK, kernel_class_80.cuh)
                                          'OUTPUT_HALVES',
                                          # an EXL3 width's bit in the widths a build decodes, a bit per bits_x2
                                          # (EXL3_WIDTH_BIT of EXL3_WIDTHS' rows, exl3_records.h)
                                          'bits_x2'),
}
SESSION_SLOT = ('slot', 'session_slot', 'row->session_slot', 'owner')
# The session-slot shifts allowed: BATCH round tables only (slot < STEP_BATCH_MAXIMUM_SESSIONS <= 32), by file.
BATCH_SITES = Counter({('source/engine/runtime/scheduler/scheduler.c', 'row->session_slot'): 1,
                       ('source/engine/runtime/scheduler/scheduler.c', 'slot'): 1,
                       ('source/engine/runtime/step/step.c', 'session_slot'): 1,
                       ('source/engine/runtime/step/step_rounds.c', 'slot'): 1,
                       ('source/engine/runtime/step/step_sessions.c', 'session_slot'): 1})
classified = {amount: kind for kind, amounts in KINDS.items() for amount in amounts}
problems, sites, session_sites = [], 0, Counter()
for directory in ('source', ):
    for path in sorted(Path(directory).rglob('*')):
        if path.suffix not in ('.c', '.h', '.cu', '.cuh'):
            continue
        clean = LITERALS.sub(lambda match: ''.join('\n' if c == '\n' else ' ' for c in match[0]), path.read_text())
        for match in SHIFT.finditer(clean):
            sites += 1
            amount, line = match[1], clean.count('\n', 0, match.start()) + 1
            if amount.isdigit() or re.fullmatch(r'\d+[uU]', amount) or match[2]:
                continue  # a constant, or the bit of a bitset word (index % its word's bits): any width
            if amount in SESSION_SLOT:
                session_sites[(str(path), amount)] += 1
            elif amount not in classified:
                problems.append(f'{path}:{line}: shift by {amount} is not classified')
        for match in BIT_READ.finditer(clean):
            if match[1] in SESSION_SLOT:
                problems.append(f'{path}:{clean.count(chr(10), 0, match.start()) + 1}: a bit read by session slot '
                                f'{match[1]} (a mask indexed by session slot)')
for site, count in sorted(session_sites.items()):
    if count > BATCH_SITES[site]:
        problems.append(f'{site[0]}: {count - BATCH_SITES[site]} session-slot shift(s) by {site[1]} outside BATCH')
for site, count in sorted(BATCH_SITES.items()):
    if session_sites[site] != count:
        problems.append(f'{site[0]}: the BATCH shift by {site[1]} is gone: update BATCH_SITES')
for problem in problems:
    print(problem)
print(f'slot width: {sites} shift sites, {sum(session_sites.values())} by session slot (BATCH only)')
sys.exit(bool(problems))
