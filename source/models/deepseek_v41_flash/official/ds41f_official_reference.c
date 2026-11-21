/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "ds41f_official_reference.h"
#include <stddef.h>
#include "models/deepseek_v41_flash/engram/ds41f_engram_rows_records.h"
#include "engine/library/bf16_linear/dense_linear_records.h"
#include "models/deepseek_v41_flash/hyper_connection/ds41f_hyper_connection_records.h"

#define OK STATUS_OK
#define THEN(call) (status == OK ? (call) : status)

MUST_CHECK static result_status
raw_kernel_prepare(const struct backend_operations *backend,
		   struct device_card *card, const char *file, const char *name,
		   uint32_t bytes, uint32_t *kernel_out)
{
	uint32_t kernel = BACKEND_NO_KERNEL;
	const result_status status =
		backend->kernel_prepare(card, file, name, bytes, &kernel);
	if (status != OK)
		return status;
	if (kernel == BACKEND_NO_KERNEL)
		return STATUS_INVALID_ARGUMENT;
	*kernel_out = kernel;
	return OK;
}

result_status ds41f_official_reference_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class,
	struct ds41f_official_reference_kernels *kernels_out)
{
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernels_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	struct ds41f_official_reference_kernels kernels = { 0 };
	result_status status = ds41f_official_linear_kernels_prepare(
		backend, card, arithmetic_class, &kernels.linear);
	status = THEN(ds41f_official_norm_kernel_prepare(backend, card,
							 &kernels.norm));
	status = THEN(ds41f_official_rope_kernel_prepare(backend, card,
							 &kernels.rope));
	status = THEN(ds41f_official_roundtrip_kernel_prepare(
		backend, card, &kernels.roundtrip));
	status = THEN(ds41f_official_head_kernel_prepare(backend, card,
							 &kernels.head));
	status = THEN(ds41f_official_router_projection_kernel_prepare(
		backend, card, &kernels.router_projection));
	status = THEN(ds41f_official_router_order_kernel_prepare(
		backend, card, &kernels.router_order));
	status = THEN(ds41f_official_attention_kernel_prepare(
		backend, card, &kernels.attention));
	status = THEN(ds41f_official_index_kernel_prepare(backend, card,
							  &kernels.index));
#define PREPARE(file, name, record, field)                                \
	status = THEN(raw_kernel_prepare(backend, card, file, name,       \
					 (uint32_t)sizeof(struct record), \
					 &kernels.field))
	PREPARE("ds41f_official_state", "official_window_state",
		ds41f_attention_prepare_arguments, window);
	PREPARE("ds41f_official_state", "official_compressed_state",
		ds41f_attention_compressor_arguments, compressor);
	PREPARE("ds41f_official_state", "official_key_state",
		ds41f_attention_key_arguments, key);
	PREPARE("ds41f_official_state", "official_index_queries",
		ds41f_attention_queries_arguments, index_queries);
	PREPARE("ds41f_official_site", "official_site_finish",
		ds41f_hyper_finish_arguments, site);
	PREPARE("ds41f_official_engram", "official_engram_gate",
		ds41f_engram_gate_arguments, engram);
	PREPARE("ds41f_official_projection", "official_projection_float32",
		dense_linear_arguments, projection_float32);
	PREPARE("ds41f_official_projection", "official_projection_bfloat16",
		dense_linear_arguments, projection_bfloat16);
	PREPARE("kernel_expert_linear", "expert_gate_up_1",
		expert_gate_up_arguments, expert_gate_up);
	PREPARE("kernel_expert_linear", "expert_down_1", expert_down_arguments,
		expert_down);
	PREPARE(DS41F_MODULE_HYPER_CONNECTION, DS41F_KERNEL_HYPER_FINAL_KEPT,
		ds41f_hyper_finish_arguments, final_kept);
#undef PREPARE
	if (status == OK)
		*kernels_out = kernels;
	return status;
}
