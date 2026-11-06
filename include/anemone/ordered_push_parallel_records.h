/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Fixed component candidate launch geometry; payload and argument ABI are ordered_push_records.h's. */
#ifndef ORDERED_PUSH_PARALLEL_RECORDS_H
#define ORDERED_PUSH_PARALLEL_RECORDS_H

#include "anemone/ordered_push_records.h"

#define KERNEL_FILE_ORDERED_PUSH_PARALLEL "kernel_ordered_push_parallel"
#define KERNEL_ORDERED_PUSH_PARALLEL_PUBLISH "ordered_push_parallel_publish"
#define KERNEL_ORDERED_PUSH_PARALLEL_REDUCE "ordered_push_parallel_reduce"
#define ORDERED_PUSH_PARALLEL_LAUNCHES 2u
#define ORDERED_PUSH_PARALLEL_BLOCKS 16u

#endif
