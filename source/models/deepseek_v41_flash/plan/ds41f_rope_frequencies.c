/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_rope_frequencies.c: the rope frequency words (split from model_v41.c by whole items).
 */
#include "model_v41.h"
#include "models/deepseek_v41_flash/official/ds41f_official_hc.h"
#include "models/deepseek_v41_flash/official/ds41f_official_reference.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_layer.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_stage.h"
#include "engine/library/exl3/exl3_tc_records.h"
#include "engine/library/rings/exact_ring_records.h"
#include "engine/library/exl3/exl3.h"
#include "models/deepseek_v41_flash/draft/ds41f_dspark_records.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_draft_share_records.h"
#include "engine/library/sampling/sampler_shard_records.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine/backend/fill_records.h"
#include "engine/library/block_scaled_linear/expert_relaxed.h"
#include "common/hooks.h"
#include "models/deepseek_v41_flash/indexer/ds41f_index_scan.h"
#include "engine/formats/digest/pack_digest.h"
#include "engine/library/block_scaled_linear/relaxed_linear.h"
#include "engine/library/prepare_checks.h"

/* The bits of both thetas' words as the reference forms them on the card (REF:376-387), recorded by
 * stepwise_capture.py --frequencies: theta 0 CFG:30 without YaRN, theta 1 CFG:51 with YaRN (factor 16,
 * original length 65,536, beta 32 and 1). The card's power lands a place below the float32 nearest
 * 10000^(2 i / 64) at words 11 and 25 of theta 0, where the host's expression gave the nearest, and the table gate
 * takes no place (first_token_plan.md 2.10). */
static const uint32_t frequency_word_bits[DS41F_THETAS][DS41F_FREQUENCIES] = {
	{ 0x3f800000u, 0x3f3ff911u, 0x3f0ff59au, 0x3ed7e89bu, 0x3ea1e89bu,
	  0x3e72d423u, 0x3e361887u, 0x3e088d77u, 0x3dcccccdu, 0x3d99940du,
	  0x3d6655c2u, 0x3d2cba16u, 0x3d0186e3u, 0x3cc2434fu, 0x3c91ad39u,
	  0x3c5a7bf2u, 0x3c23d70au, 0x3bf5b9b0u, 0x3bb8449cu, 0x3b8a2e77u,
	  0x3b4f3e38u, 0x3b1b690du, 0x3ae91528u, 0x3aaec98eu, 0x3a83126fu,
	  0x3a44948du, 0x3a136a16u, 0x39dd1725u, 0x39a5cb60u, 0x3978a815u,
	  0x393a7753u, 0x390bd472u },
	{ 0x3f800000u, 0x3f300a3au, 0x3ef21c1fu, 0x3ea67d01u, 0x3e64f92eu,
	  0x3e1d7475u, 0x3dd88cb4u, 0x3d94e963u, 0x3d4ccccdu, 0x3d0cd4fbu,
	  0x3cc1b019u, 0x3c8530ceu, 0x3c372dbfu, 0x3bfbed88u, 0x3bad3d5eu,
	  0x3b6e4237u, 0x3b147ae1u, 0x3ab714e0u, 0x3a5ebdb6u, 0x3a0530ceu,
	  0x399bb3afu, 0x39305978u, 0x38be904du, 0x383e9b5fu, 0x37a3d708u,
	  0x36b443d0u, 0x3677eba6u, 0x362a7be8u, 0x35ea77ffu, 0x35a13bdcu,
	  0x355dbf30u, 0x35187c4du },
};
_Static_assert(sizeof(frequency_word_bits) ==
		       sizeof(float[DS41F_THETAS][DS41F_FREQUENCIES]),
	       "a word's bits are a float32");

void ds41f_frequency_words(float words_out[DS41F_THETAS][DS41F_FREQUENCIES])
{
	memcpy(words_out, frequency_word_bits, sizeof(frequency_word_bits));
}
