/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef OFFICIAL_HC_RECORDS_H
#define OFFICIAL_HC_RECORDS_H

#include "anemone/dense_linear_records.h"
#include "anemone/model_layout.h"

/* Reuses dense_linear_arguments, with the restricted contract in official_hc.h. */
#define OFFICIAL_HC_THREADS 512u
#define OFFICIAL_HC_INPUTS (MODEL_HIDDEN * MODEL_STREAMS)
#define OFFICIAL_HC_ODD_PROMPT_ROWS 3u
#define KERNEL_FILE_OFFICIAL_HC "kernel_official_hc"
#define KERNEL_OFFICIAL_HC_DOT "official_hc_dot47"
#define KERNEL_OFFICIAL_HC_SPLIT "official_hc_split128"
#define KERNEL_OFFICIAL_HC_STAGED "official_hc_split128_staged"

#endif
