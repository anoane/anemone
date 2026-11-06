/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * site_front.h: the fused fronts of a block's sites on the cards (site_front_records.h): the site front, one launch in
 * place of a site's mix project, mix finish and the quantization behind it (model_v41_program.c: site_begin, then
 * quantize: the attention site's ahead of attn.wq_a with attn.wkv), and the FFN front, one launch in place of the FFN
 * site's mix project, mix finish, router scores, router select and quantization (ffn_launches up to the routed
 * experts). The site front may also be cut in two launches, the input front and then the coefficient front, which may
 * run beside the launches up to the site's post (site_front_records.h). Each is prepared once at load from a
 * description that names every arena offset.
 *
 * Contract: linear.h's for the kernels and the prepare functions (once at load, arithmetic on a description, one error
 * line for a refusal, the launch untouched). Refused with STATUS_INVALID_ARGUMENT: a NULL argument (router
 * and router_kernels are NULL together, exactly where kind is not SITE_FRONT_KIND_FFN); a kind outside the
 * enumeration; a kernel it launches not prepared; a reserved field that is not zero; a row_count of 0 or above
 * LINEAR_MAXIMUM_ROWS, or not the routed experts' row_count; an FFN router other than the model's
 * 384 experts and six picks (the draft uses the site front and its separate router); a range
 * that breaks a rule of prepare_common.h for the bytes its comment gives, of row_count rows of model_layout.h's arrays
 * ("exactly": a tensor of the pack, which holds exactly its bytes); and, for the FFN front, whatever
 * router_select_prepare (router.h) refuses of router, which it runs first: its ranges then also share no byte
 * with the description's.
 */
#ifndef SITE_FRONT_H
#define SITE_FRONT_H

#include <stdint.h>

#include "anemone/backend_seam.h"
#include "anemone/linear.h"
#include "anemone/router.h"
#include "anemone/site_front_records.h"
#include "anemone/status.h"

/* The handles of the functions of kernel_site_front on one card, in the order of the stamps of site_front_records.h. */
struct site_front_kernels {
	uint32_t fronts[SITE_FRONT_INSTANCES];
};
/* Prepares them on one card, as linear_kernels_prepare does. */
MUST_CHECK result_status site_front_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	struct site_front_kernels *kernels_out);

struct site_front_description {
	struct linear_range streams; /* read: streams */
	struct linear_range
		carried_pre; /* read, then written: the previous site's pre, then this site's */
	struct linear_range
		mix_weights; /* read: hc_attn_fn or hc_ffn_fn, float32 [24][20,480], exactly */
	struct linear_range
		scale; /* read: hc_attn_scale or hc_ffn_scale, exactly */
	struct linear_range base; /* read: hc_attn_base or hc_ffn_base, exactly */
	struct linear_range
		norm_weight; /* read: attn_norm.weight or ffn_norm.weight, exactly */
	struct linear_range post; /* written: post */
	struct linear_range comb; /* written: comb */
	struct linear_range hidden; /* written: hidden, the sublayer's input */
	struct linear_range
		codes; /* written: codes [row_count][5,120] (linear_records.h) */
	struct linear_range exponents; /* written: exponents [row_count][160] */
	struct linear_range operands; /* written: operands [row_count][5,120] */
	struct linear_range
		scratch; /* read and written: struct site_front_scratch, counters zero */
	const struct router_description *
		router; /* the FFN front's select (its scores written too), or NULL */
	struct linear_range
		router_weights; /* read by the FFN front: ffn.gate.weight [384][5,120], exactly */
	uint32_t kind; /* enum site_front_kind; FFN exactly where router is given */
	uint32_t row_count;
	uint32_t reserved_zero;
};
MUST_CHECK result_status
site_front_prepare(const struct site_front_description *description,
		   uint64_t arena_byte_count,
		   const struct router_kernels_record *router_kernels,
		   const struct site_front_kernels *kernels,
		   struct kernel_launch *launch_out);

#endif
