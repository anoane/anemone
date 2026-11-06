/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Shared NVMe read-bound policy for both storage implementations. Zero means
 * unknown, with the storage seam's accepted warning policy. No card is touched. */
#ifndef STORAGE_FILE_BOUNDS_H
#define STORAGE_FILE_BOUNDS_H
#include <stdint.h>
uint64_t storage_drive_read_bound_milliseconds(uint32_t device_major,
					       uint32_t device_minor);
#endif
