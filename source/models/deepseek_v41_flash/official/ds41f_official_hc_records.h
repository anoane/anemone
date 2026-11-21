/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef DS41F_OFFICIAL_HC_RECORDS_H
#define DS41F_OFFICIAL_HC_RECORDS_H

#include "engine/library/bf16_linear/dense_linear_records.h"
#include "models/deepseek_v41_flash/ds41f_geometry.h"

/* Reuses dense_linear_arguments, with the restricted contract in ds41f_official_hc.h. */
#define DS41F_OFFICIAL_HC_THREADS 512u
#define DS41F_OFFICIAL_HC_INPUTS (DS41F_HIDDEN * DS41F_STREAMS)
#define DS41F_OFFICIAL_HC_ODD_PROMPT_ROWS 3u
#define DS41F_MODULE_OFFICIAL_HC "ds41f_official_hc"
#define DS41F_KERNEL_OFFICIAL_HC_DOT "official_hc_dot47"
#define DS41F_KERNEL_OFFICIAL_HC_SPLIT "official_hc_split128"
#define DS41F_KERNEL_OFFICIAL_HC_STAGED "official_hc_split128_staged"

#endif
