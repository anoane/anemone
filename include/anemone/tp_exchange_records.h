/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* TP-only phase and active-row adaptation of the existing ordered push protocol. */
#ifndef TP_EXCHANGE_RECORDS_H
#define TP_EXCHANGE_RECORDS_H

#include "anemone/model_layout.h"
#include "anemone/linear_records.h"
#include "anemone/ordered_push_records.h"

#define KERNEL_FILE_TP_EXCHANGE "kernel_tp_exchange"
#define KERNEL_TP_EXCHANGE "tp_exchange"
/* tp_exchange.h's wide form: four launches, the first and third over TP_EXCHANGE_WIDE_BLOCKS CTAs */
#define KERNEL_TP_EXCHANGE_PUBLISH "tp_exchange_publish"
#define KERNEL_TP_EXCHANGE_READY "tp_exchange_ready"
#define KERNEL_TP_EXCHANGE_SUM "tp_exchange_sum"
#define KERNEL_TP_EXCHANGE_DONE "tp_exchange_done"
/* the owner sums' and owner ring sums' PUBLISH and SUM (tp_exchange.h) */
#define KERNEL_TP_EXCHANGE_OWNER_PUBLISH "tp_exchange_owner_publish"
#define KERNEL_TP_EXCHANGE_OWNER_SUM "tp_exchange_owner_sum"
#define KERNEL_TP_EXCHANGE_OWNER_RING_PUBLISH "tp_exchange_owner_ring_publish"
#define KERNEL_TP_EXCHANGE_OWNER_RING_SUM "tp_exchange_owner_ring_sum"
#define TP_EXCHANGE_FORM_PUSH 0u
#define TP_EXCHANGE_FORM_OWNER_SUMS 1u
#define TP_EXCHANGE_FORM_OWNER_RING_SUMS 2u
#define TP_EXCHANGE_FORM_CE_OWNER_SUMS \
	3u /* the owner sums, the bytes moved by copy engines (no PUBLISH) */
#define KERNEL_TP_EXCHANGE_CE_OWNER_SUM "tp_exchange_ce_owner_sum"
#define TP_EXCHANGE_FORM_CE_OWNER_RING_SUMS \
	4u /* the owner ring sums by copy engines (no PUBLISH) */
#define KERNEL_TP_EXCHANGE_CE_OWNER_RING_SUM "tp_exchange_ce_owner_ring_sum"
#define TP_EXCHANGE_OWNER_RING_RANKS \
	4u /* the BF16 ring's ranks (kernel_cut_through.cuh) */
#define TP_EXCHANGE_OWNER_ALIGNMENT 16u
#define TP_EXCHANGE_WIDE_LAUNCHES 4u
/* Phases a layer: an exchange's layer argument is 8 l + 2 + 2 h, h the half of split_overlap's chunk (0 without it),
 * its pipeline's gathers at 8 l + 0, 1 and 5, each where no other exchange of its channel runs beside it
 * (model_v41_program.c), and its layer count TP_EXCHANGE_PHASES for every exchange of every program, so a
 * region's phases keep increasing. */
#define TP_EXCHANGE_LAYER_PHASES 8u
#define TP_EXCHANGE_PHASES (TP_EXCHANGE_LAYER_PHASES * MODEL_LAYERS)
#define TP_EXCHANGE_WIDE_BLOCKS 64u
#define TP_EXCHANGE_ARGUMENT_BYTES 112u
/* a channel's rows: split_overlap's chunks of two wide launches (model_v41_tp2_stage.h), an exchange a half */
#define TP_EXCHANGE_MAXIMUM_ELEMENTS (MODEL_HIDDEN * 2u * LINEAR_WIDE_MOST_ROWS)

struct tp_exchange_arguments {
	struct ordered_push_arguments collective;
	uint32_t layer;
	uint32_t layer_count;
	uint64_t reserved_zero;
};

MODEL_LAYOUT_HOLDS(sizeof(struct tp_exchange_arguments) ==
		   TP_EXCHANGE_ARGUMENT_BYTES);

#endif
