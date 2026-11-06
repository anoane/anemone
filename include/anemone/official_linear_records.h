/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Original packed E4M3 projections. The launch records are the retained
 * 72-byte quantized and 40-byte grouped-BF16 trunk records, without taps. */
#ifndef OFFICIAL_LINEAR_RECORDS_H
#define OFFICIAL_LINEAR_RECORDS_H

#include "anemone/linear_records.h"

#define KERNEL_FILE_OFFICIAL_LINEAR "kernel_official_linear"
#define KERNEL_OFFICIAL_LINEAR "official_linear"
#define KERNEL_OFFICIAL_LINEAR_NATIVE "official_linear_native"
#define KERNEL_FILE_OFFICIAL_LINEAR_WOA "kernel_official_linear_woa"
#define KERNEL_OFFICIAL_LINEAR_WOA "official_linear_woa"
#define OFFICIAL_LINEAR_THREADS 128u
#define OFFICIAL_LINEAR_PORTABLE_OUTPUTS 4u
#define OFFICIAL_LINEAR_NATIVE_OUTPUTS 32u
#define OFFICIAL_LINEAR_NATIVE_ROWS 16u
#define OFFICIAL_LINEAR_WOA_OUTPUTS 8u
#define OFFICIAL_LINEAR_WOA_GROUPS 8u
#define OFFICIAL_LINEAR_WOA_COLUMNS 1024u
#define OFFICIAL_LINEAR_WOA_INNER 4096u

#endif
