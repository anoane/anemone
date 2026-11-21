/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Serial original state producers: BF16 window rings and index keys/queries;
 * group16/E4 pools retain lossless FP16 carriers. Ordinary records are retained. */
#ifndef DS41F_OFFICIAL_STATE_RECORDS_H
#define DS41F_OFFICIAL_STATE_RECORDS_H
#include "models/deepseek_v41_flash/attention/ds41f_attention_state_records.h"
#define DS41F_OFFICIAL_STATE_THREADS 128u
#define DS41F_OFFICIAL_SITE_THREADS_VALUE 544u
#define DS41F_OFFICIAL_PROJECTION_OWNERS_VALUE 32u
#define DS41F_OFFICIAL_PROJECTION_OUTPUTS_VALUE 4u
#endif
