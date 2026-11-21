/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Original packed E4M3 projections. The launch records are the retained
 * 72-byte quantized and 40-byte grouped-BF16 trunk records, without taps. */
#ifndef DS41F_OFFICIAL_LINEAR_RECORDS_H
#define DS41F_OFFICIAL_LINEAR_RECORDS_H

#include "engine/library/block_scaled_linear/linear_records.h"

#define DS41F_MODULE_OFFICIAL_LINEAR "ds41f_official_linear"
#define DS41F_KERNEL_OFFICIAL_LINEAR "official_linear"
#define DS41F_KERNEL_OFFICIAL_LINEAR_NATIVE "official_linear_native"
#define DS41F_MODULE_OFFICIAL_LINEAR_WOA "ds41f_official_linear_woa"
#define DS41F_KERNEL_OFFICIAL_LINEAR_WOA "official_linear_woa"
#define DS41F_OFFICIAL_LINEAR_THREADS 128u
#define DS41F_OFFICIAL_LINEAR_PORTABLE_OUTPUTS 4u
#define DS41F_OFFICIAL_LINEAR_NATIVE_OUTPUTS 32u
#define DS41F_OFFICIAL_LINEAR_NATIVE_ROWS 16u
#define DS41F_OFFICIAL_LINEAR_WOA_OUTPUTS 8u
#define DS41F_OFFICIAL_LINEAR_WOA_GROUPS 8u
#define DS41F_OFFICIAL_LINEAR_WOA_COLUMNS 1024u
#define DS41F_OFFICIAL_LINEAR_WOA_INNER 4096u

#endif
