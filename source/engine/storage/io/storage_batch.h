/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef STORAGE_BATCH_H
#define STORAGE_BATCH_H
#include "engine/storage/storage_seam.h"
#include "engine/formats/pack/pack.h"
#define STORAGE_BATCH_DEPTH 32u
struct storage_batch;
/* Prepared direct I/O with registered copy descriptors and bounded ring-owned buffers. A timeout quarantines
 * that ring's mappings/buffers until process exit; no outstanding I/O ever references caller-owned memory. */
MUST_CHECK result_status storage_batch_open(const struct storage_reader *reader,
					    struct storage_batch **batch_out);
void storage_batch_close(struct storage_batch *batch);
MUST_CHECK result_status storage_batch_read(struct storage_batch *batch,
					    const uint32_t *copies,
					    const uint64_t *offsets,
					    uint32_t count, void *blocks_out,
					    result_status *results_out);
/* For storage_flight_linux.c: a copy's descriptor, and the physical offset of a logical range wholly inside one
 * mapped table (or the logical offset itself for a full-pack reader); false for a range the reader does not hold. */
int storage_direct_descriptor(const struct storage_reader *reader,
			      uint32_t copy);
bool storage_direct_physical_offset(const struct storage_reader *reader,
				    uint32_t copy, uint64_t logical,
				    uint64_t bytes, uint64_t *physical_out);
#endif
