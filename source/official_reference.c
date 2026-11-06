/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/official_reference.h"
#include <stddef.h>
#include "anemone/engram_rows_records.h"
#include "anemone/dense_linear_records.h"
#include "anemone/hyper_connection_records.h"

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

result_status official_reference_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	uint32_t arithmetic_class,
	struct official_reference_kernels *kernels_out)
{
	if (backend == NULL || backend->kernel_prepare == NULL ||
	    card == NULL || kernels_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	struct official_reference_kernels kernels = { 0 };
	result_status status = official_linear_kernels_prepare(
		backend, card, arithmetic_class, &kernels.linear);
	status = THEN(
		official_norm_kernel_prepare(backend, card, &kernels.norm));
	status = THEN(
		official_rope_kernel_prepare(backend, card, &kernels.rope));
	status = THEN(official_roundtrip_kernel_prepare(backend, card,
							&kernels.roundtrip));
	status = THEN(
		official_head_kernel_prepare(backend, card, &kernels.head));
	status = THEN(official_router_projection_kernel_prepare(
		backend, card, &kernels.router_projection));
	status = THEN(official_router_order_kernel_prepare(
		backend, card, &kernels.router_order));
	status = THEN(official_attention_kernel_prepare(backend, card,
							&kernels.attention));
	status = THEN(
		official_index_kernel_prepare(backend, card, &kernels.index));
#define PREPARE(file, name, record, field)                                \
	status = THEN(raw_kernel_prepare(backend, card, file, name,       \
					 (uint32_t)sizeof(struct record), \
					 &kernels.field))
	PREPARE("kernel_official_state", "official_window_state",
		attention_prepare_arguments, window);
	PREPARE("kernel_official_state", "official_compressed_state",
		attention_compressor_arguments, compressor);
	PREPARE("kernel_official_state", "official_key_state",
		attention_key_arguments, key);
	PREPARE("kernel_official_state", "official_index_queries",
		attention_queries_arguments, index_queries);
	PREPARE("kernel_official_site", "official_site_finish",
		hyper_finish_arguments, site);
	PREPARE("kernel_official_engram", "official_engram_gate",
		engram_gate_arguments, engram);
	PREPARE("kernel_official_projection", "official_projection_float32",
		dense_linear_arguments, projection_float32);
	PREPARE("kernel_official_projection", "official_projection_bfloat16",
		dense_linear_arguments, projection_bfloat16);
	PREPARE("kernel_expert_linear", "expert_gate_up_1",
		expert_gate_up_arguments, expert_gate_up);
	PREPARE("kernel_expert_linear", "expert_down_1", expert_down_arguments,
		expert_down);
	PREPARE(KERNEL_FILE_HYPER_CONNECTION, KERNEL_HYPER_FINAL_KEPT,
		hyper_finish_arguments, final_kept);
#undef PREPARE
	if (status == OK)
		*kernels_out = kernels;
	return status;
}
