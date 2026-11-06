/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_exl3_tcp.cu: the routed experts of one EXL3 layer on the tensor cores for prefill-sized calls (more than 64
 * rows), from the router's picks, one TP rank's intermediate slice (exl3_tc_records.h; arithmetic: kernel_exl3_tcp.cuh,
 * Morrowmake's `moe_out_prefill_try`, GNmac's prefill MoE). Device code only, sm_80, module "kernel_exl3_tcp". The
 * GEMMs keep one block a multiprocessor (255 registers), or (_b2, --exl3_tcp_two_blocks) two (128 registers, the
 * k16 steps not unrolled): no spill, each output's MMA and add order the same. Kernels:
 *   route: one block, the pairs grouped by local expert into 64-row tiles, the experts by decreasing rows;
 *   retile (half_rows, split_overlap mode 7): one block, the route's tiles a half each for the down GEMMs;
 *   operands: a block a (sorted position, projection) task;
 *   gemm_{gate_up,gate,up}_i{640,512}_k{3,2}, gemm_down_i{640,512}_k{3,2}: a block an item (tile, projection, column
 *     block);
 *   down_i{640,512}: a warp a sorted position;
 *   combine: a warp a (row, 512 outputs), FP32 out.
 * --exl3_expert_rotation (record.rotation, intermediate 640): each expert 640 or 512 wide (exl3_tc_width), the
 * 640 layout throughout; gemm_*_m640_k{3,2} and down_m640 run each tile or position at its expert's width, bitwise the
 * i640 / i512 forms' per route; route, operands and combine read the width with every record.
 */
#include "anemone/exl3_tc_records.h"
#include "anemone/model_layout.h"
#include "anemone/router_records.h"
#include "anemone/step_records.h"
#include "kernel_exl3_tcp.cuh"

#define PAIRS_PER_ROW EXL3_TC_PAIRS_PER_ROW
static_assert(EXL3_TCP_TILE_ROWS == TCP_TILE_ROWS &&
		      EXL3_TCP_JOINT_ROWS == TCP_JOINT_ROWS &&
		      EXL3_TCP_CHUNK_ROWS == TCP_CHUNK_ROWS &&
		      EXL3_TCP_GEMM_THREADS == TCP_GEMM_THREADS &&
		      EXL3_TCP_OPERAND_THREADS == 32 * (TC_MOE_HIDDEN / 1024) &&
		      EXL3_TCP_DOWN_THREADS == 32 * TCP_DOWN_WARPS,
	      "one geometry");

struct tcp_parts { /* the record's workspace (exl3_tc_records.h) */
	__half *operand;
	float *down_result, *gate_result, *up_result, *unscale;
	int32_t *table;
	int64_t pairs, projections;
};
__device__ __forceinline__ struct tcp_parts
tcp_parts_of(const struct exl3_tcp_record &record, unsigned char *arena)
{
	struct tcp_parts parts;
	parts.pairs = (int64_t)record.row_count * PAIRS_PER_ROW;
	parts.projections = record.row_count <= TCP_JOINT_ROWS ? 2 : 1;
	float *base = (float *)(arena + record.workspace_offset);
	parts.operand = (__half *)base;
	parts.down_result = base;
	parts.gate_result = base + parts.pairs * parts.projections * TC_MOE_HIDDEN;
	parts.up_result = parts.gate_result + parts.pairs * record.intermediate;
	parts.unscale = parts.up_result + parts.pairs * record.intermediate;
	parts.table = (int32_t *)(parts.unscale + (parts.pairs * parts.projections + 3) / 4 * 4);
	return parts;
}
__device__ __forceinline__ const struct router_pick &
tcp_pick(const struct exl3_tcp_record &record, unsigned char *arena,
	 int64_t pair)
{
	return ((const struct router_pick *)(arena +
					     record.picks_offset))[pair];
}
/* An expert's matrix (0 gate, 1 up, 2 down): its trellis, then FP16 suh[input], svh[output]; the expert `width`
 * wide (tcp_width). */
__device__ __forceinline__ uint64_t
tcp_matrix(const struct exl3_tcp_record &record, unsigned char *arena,
	   int expert, int projection)
{
	return ((const uint64_t *)(arena + record.expert_offsets_offset))[expert * 3 + projection];
}
__device__ __forceinline__ uint32_t
tcp_width(const struct exl3_tcp_record &record, int expert)
{
	return exl3_tc_width(record.intermediate, record.rotation,
			     (uint32_t)expert);
}
__device__ __forceinline__ uint64_t tcp_suh(const struct exl3_tcp_record &record,
					    int projection, uint64_t matrix,
					    uint32_t width)
{
	const uint64_t input = projection < 2 ? TC_MOE_HIDDEN : width;
	const uint64_t output = projection < 2 ? width : TC_MOE_HIDDEN;
	return matrix + input * output * record.bits_x2 / 16u;
}
__device__ __forceinline__ uint64_t tcp_svh(const struct exl3_tcp_record &record,
					    int projection, uint64_t matrix,
					    uint32_t width)
{
	return tcp_suh(record, projection, matrix, width) + (projection < 2 ? TC_MOE_HIDDEN : width) * sizeof(__half);
}

/* A pair's local expert, or -1: a full step's row whose output at the record's decoder layer nothing reads takes no
 * route (cone_layer, the cone); half (half_rows): its route expert 2 e + h, h its row's half (tcp_route_paired). */
__device__ __forceinline__ int
tcp_route_of(const struct exl3_tcp_record &record, unsigned char *arena,
	     int64_t pair, bool half)
{
	const struct step_row_record *const rows =
		(const struct step_row_record *)(arena + record.step_offset +
						 STEP_BLOCK_BYTES_VALUE);
	const uint32_t expert = tcp_pick(record, arena, pair).expert;
	const uint32_t flags = record.cone_layer != 0 ? rows[pair / PAIRS_PER_ROW].flags : 0u;
	if (expert >= EXL3_TCP_EXPERTS ||
	    STEP_ROW_CONE_SKIPS(flags,
				record.cone_layer - MODEL_DECODER_FIRST_LAYER))
		return -1;
	return half ? TCP_HALVES * (int)expert + (pair / PAIRS_PER_ROW >= record.half_rows ? 1 : 0) : (int)expert;
}

extern "C" __global__ void __launch_bounds__(TCP_ROUTE_THREADS)
	exl3_tcp_route(struct exl3_tcp_record record, unsigned char *arena)
{
	const struct tcp_parts parts = tcp_parts_of(record, arena);
	for (int expert = threadIdx.x; expert < (int)EXL3_TCP_EXPERTS;
	     expert += blockDim.x)
		for (int projection = 0; projection < 3; projection++)
			if ((tcp_matrix(record, arena, expert, projection) & 15) != 0)
				__trap();
	/* half_rows: the gate/up tiles over both halves (tcp_route_paired), the halves' down tiles by retile */
	if (record.half_rows != 0)
		tcp_route_paired<false>(
			parts.pairs, [&](int64_t pair) { return tcp_route_of(record, arena, pair, true); },
			parts.table);
	else
		tcp_route(
			parts.pairs, (int)EXL3_TCP_EXPERTS, 1,
			[&](int64_t pair) {
				return tcp_route_of(record, arena, pair, false);
			},
			parts.table);
}

/* half_rows, behind the gate/up GEMMs: the route table's tiles become each half's down tiles (tcp_route_paired). */
extern "C" __global__ void __launch_bounds__(TCP_ROUTE_THREADS)
	exl3_tcp_retile(struct exl3_tcp_record record, unsigned char *arena)
{
	const struct tcp_parts parts = tcp_parts_of(record, arena);
	tcp_route_paired<true>(
		parts.pairs, [&](int64_t pair) { return tcp_route_of(record, arena, pair, true); }, parts.table);
}

extern "C" __global__ void __launch_bounds__(EXL3_TCP_OPERAND_THREADS)
	exl3_tcp_operands(struct exl3_tcp_record record, unsigned char *arena)
{
	const struct tcp_parts parts = tcp_parts_of(record, arena);
	const int first = record.pass == EXL3_TCP_UP ? 1 : 0;
	const int64_t projections = record.pass == EXL3_TCP_GATE_AND_UP ? 2 : 1;
	const int32_t *sorted = parts.table + 4;
	const int64_t tasks = (int64_t)parts.table[0] * projections;
	for (int64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
		const int64_t position = task / projections;
		const int projection = first + (int)(task % projections);
		const int64_t pair = sorted[position];
		const int expert = (int)tcp_pick(record, arena, pair).expert;
		tcp_operand_row<TC_MOE_HIDDEN>(
			(const __nv_bfloat16 *)(arena + record.hidden_offset) + (pair / PAIRS_PER_ROW) * TC_MOE_HIDDEN,
			(const __half *)(arena + tcp_suh(record, projection,
							 tcp_matrix(record, arena, expert, projection),
							 tcp_width(record, expert))),
			parts.operand + task * 2 * TC_MOE_HIDDEN, parts.unscale + task);
	}
}

/* A GEMM launch's parameters: the record's pass reads the gate/up operands or the down operand (the gate results'
 * region) and writes its projections' results. */
__device__ __forceinline__ struct tcp_gemm
tcp_gemm_of(const struct exl3_tcp_record &record, unsigned char *arena)
{
	const struct tcp_parts parts = tcp_parts_of(record, arena);
	const bool down = record.pass == EXL3_TCP_DOWN;
	struct tcp_gemm p;
	p.table = parts.table;
	p.pairs = parts.pairs;
	p.experts = EXL3_TCP_EXPERTS;
	p.operand = down ? (const __half *)parts.gate_result : parts.operand;
	p.unscale = parts.unscale;
	p.out = down			   ? parts.down_result :
		record.pass == EXL3_TCP_UP ? parts.up_result :
					     parts.gate_result;
	p.out_projection = down ? 0 : parts.up_result - parts.gate_result;
	/* a part's tiles: the first half's come first (tcp_route's halves) */
	p.first_tile = record.part == EXL3_TCP_SECOND_HALF ? parts.table[2] : 0;
	p.end_tile = record.part == EXL3_TCP_FIRST_HALF ? parts.table[2] :
							  parts.table[1];
	return p;
}

/* A route tile's expert: its route expert less the experts of a second half's (tcp_route's halves). */
__device__ __forceinline__ int tcp_route_expert(int route_expert)
{
	return route_expert % (int)EXL3_TCP_EXPERTS;
}

template <typename shape>
__device__ __forceinline__ void
tcp_gemm_launch(const struct exl3_tcp_record &record, unsigned char *arena)
{
	tcp_gemm_item<shape>(
		tcp_gemm_of(record, arena),
		[&](int expert, int projection) {
			return (const unsigned char *)(arena + tcp_matrix(record, arena, tcp_route_expert(expert),
									  projection));
		},
		[&](int expert) { return (int)tcp_width(record, tcp_route_expert(expert)); });
}

template <typename wide, typename narrow>
__device__ __forceinline__ void
tcp_gemm_hybrid_launch(const struct exl3_tcp_record &record,
		       unsigned char *arena)
{
	tcp_gemm_hybrid_item<wide, narrow>(
		tcp_gemm_of(record, arena),
		[&](int expert, int projection) {
			return (const unsigned char *)(arena + tcp_matrix(record, arena, tcp_route_expert(expert),
									  projection));
		},
		[&](int expert) { return (int)tcp_width(record, tcp_route_expert(expert)); });
}

/* --exl3_tcp_two_blocks: each form also at two blocks a multiprocessor (128 registers, compact k16 steps: no spill),
 * named with _b2; the same MMA and add order for every output */
#define TCP_GEMM_TWO_BLOCKS 2
#define TCP_GEMM(name, shape, blocks)                                          \
	extern "C" __global__ void __launch_bounds__(TCP_GEMM_THREADS, blocks) \
		name(struct exl3_tcp_record record, unsigned char *arena)      \
	{                                                                      \
		tcp_gemm_launch<shape>(record, arena);                         \
	}
#define TCP_GATE_UP(inner, bits, two) tcp_gate_up_shape<TC_MOE_HIDDEN, inner, bits, 2, 0, false, two>
#define TCP_GATE(inner, bits, two) tcp_gate_up_shape<TC_MOE_HIDDEN, inner, bits, 1, 0, false, two>
#define TCP_UP(inner, bits, two) tcp_gate_up_shape<TC_MOE_HIDDEN, inner, bits, 1, 1, false, two>
#define TCP_DOWN(inner, bits, two) tcp_down_shape<TC_MOE_HIDDEN, inner, bits, false, two>
/* --exl3_expert_rotation: experts of 640 or 512 columns at the 640 shapes */
#define TCP_GATE_UP_MIXED(bits, two) tcp_gate_up_shape<TC_MOE_HIDDEN, 640, bits, 2, 0, true, two>
#define TCP_GATE_MIXED(bits, two) tcp_gate_up_shape<TC_MOE_HIDDEN, 640, bits, 1, 0, true, two>
#define TCP_UP_MIXED(bits, two) tcp_gate_up_shape<TC_MOE_HIDDEN, 640, bits, 1, 1, true, two>
#define TCP_DOWN_MIXED(bits, two) tcp_down_shape<TC_MOE_HIDDEN, 640, bits, true, two>
/* the 640 hybrid (tcp_gemm_hybrid_item): two 256-column blocks of two tiles a warp, then 128 columns of one */
#define TCP_GEMM_HYBRID(name, projections, first, bits, two, blocks)           \
	extern "C" __global__ void __launch_bounds__(TCP_GEMM_THREADS, blocks) \
		name(struct exl3_tcp_record record, unsigned char *arena)      \
	{                                                                      \
		tcp_gemm_hybrid_launch<                                        \
			tcp_shape<TC_MOE_HIDDEN, 640, projections, first, 2,   \
				  bits, false, true, two>,                     \
			tcp_shape<TC_MOE_HIDDEN, 640, projections, first, 1,   \
				  bits, false, false, two> >(record, arena);   \
	}
#define TCP_GEMM_FORMS(suffix, two, blocks)                                  \
	TCP_GEMM(exl3_tcp_gemm_gate_up_i640_k3##suffix,                      \
		 TCP_GATE_UP(640, 6, two), blocks)                           \
	TCP_GEMM(exl3_tcp_gemm_gate_i640_k3##suffix, TCP_GATE(640, 6, two),  \
		 blocks)                                                     \
	TCP_GEMM(exl3_tcp_gemm_up_i640_k3##suffix, TCP_UP(640, 6, two),      \
		 blocks)                                                     \
	TCP_GEMM(exl3_tcp_gemm_down_i640_k3##suffix, TCP_DOWN(640, 6, two),  \
		 blocks)                                                     \
	TCP_GEMM(exl3_tcp_gemm_gate_up_i512_k3##suffix,                      \
		 TCP_GATE_UP(512, 6, two), blocks)                           \
	TCP_GEMM(exl3_tcp_gemm_gate_i512_k3##suffix, TCP_GATE(512, 6, two),  \
		 blocks)                                                     \
	TCP_GEMM(exl3_tcp_gemm_up_i512_k3##suffix, TCP_UP(512, 6, two),      \
		 blocks)                                                     \
	TCP_GEMM(exl3_tcp_gemm_down_i512_k3##suffix, TCP_DOWN(512, 6, two),  \
		 blocks)                                                     \
	TCP_GEMM(exl3_tcp_gemm_gate_up_i640_k2##suffix,                      \
		 TCP_GATE_UP(640, 4, two), blocks)                           \
	TCP_GEMM(exl3_tcp_gemm_gate_i640_k2##suffix, TCP_GATE(640, 4, two),  \
		 blocks)                                                     \
	TCP_GEMM(exl3_tcp_gemm_up_i640_k2##suffix, TCP_UP(640, 4, two),      \
		 blocks)                                                     \
	TCP_GEMM(exl3_tcp_gemm_down_i640_k2##suffix, TCP_DOWN(640, 4, two),  \
		 blocks)                                                     \
	TCP_GEMM(exl3_tcp_gemm_gate_up_i512_k2##suffix,                      \
		 TCP_GATE_UP(512, 4, two), blocks)                           \
	TCP_GEMM(exl3_tcp_gemm_gate_i512_k2##suffix, TCP_GATE(512, 4, two),  \
		 blocks)                                                     \
	TCP_GEMM(exl3_tcp_gemm_up_i512_k2##suffix, TCP_UP(512, 4, two),      \
		 blocks)                                                     \
	TCP_GEMM(exl3_tcp_gemm_down_i512_k2##suffix, TCP_DOWN(512, 4, two),  \
		 blocks)                                                     \
	TCP_GEMM(exl3_tcp_gemm_gate_up_m640_k3##suffix,                      \
		 TCP_GATE_UP_MIXED(6, two), blocks)                          \
	TCP_GEMM(exl3_tcp_gemm_gate_m640_k3##suffix, TCP_GATE_MIXED(6, two), \
		 blocks)                                                     \
	TCP_GEMM(exl3_tcp_gemm_up_m640_k3##suffix, TCP_UP_MIXED(6, two),     \
		 blocks)                                                     \
	TCP_GEMM(exl3_tcp_gemm_down_m640_k3##suffix, TCP_DOWN_MIXED(6, two), \
		 blocks)                                                     \
	TCP_GEMM(exl3_tcp_gemm_gate_up_m640_k2##suffix,                      \
		 TCP_GATE_UP_MIXED(4, two), blocks)                          \
	TCP_GEMM(exl3_tcp_gemm_gate_m640_k2##suffix, TCP_GATE_MIXED(4, two), \
		 blocks)                                                     \
	TCP_GEMM(exl3_tcp_gemm_up_m640_k2##suffix, TCP_UP_MIXED(4, two),     \
		 blocks)                                                     \
	TCP_GEMM(exl3_tcp_gemm_down_m640_k2##suffix, TCP_DOWN_MIXED(4, two), \
		 blocks)                                                     \
	TCP_GEMM_HYBRID(exl3_tcp_gemm_gate_up_h640_k3##suffix, 2, 0, 6, two, \
			blocks)                                              \
	TCP_GEMM_HYBRID(exl3_tcp_gemm_gate_h640_k3##suffix, 1, 0, 6, two,    \
			blocks)                                              \
	TCP_GEMM_HYBRID(exl3_tcp_gemm_up_h640_k3##suffix, 1, 1, 6, two,      \
			blocks)                                              \
	TCP_GEMM_HYBRID(exl3_tcp_gemm_gate_up_h640_k2##suffix, 2, 0, 4, two, \
			blocks)                                              \
	TCP_GEMM_HYBRID(exl3_tcp_gemm_gate_h640_k2##suffix, 1, 0, 4, two,    \
			blocks)                                              \
	TCP_GEMM_HYBRID(exl3_tcp_gemm_up_h640_k2##suffix, 1, 1, 4, two, blocks)
TCP_GEMM_FORMS(, false, 1)
TCP_GEMM_FORMS(_b2, true, TCP_GEMM_TWO_BLOCKS)
#undef TCP_GEMM_FORMS
#undef TCP_GEMM_HYBRID
#undef TCP_GEMM
#undef TCP_GATE_UP
#undef TCP_GATE
#undef TCP_UP
#undef TCP_DOWN
#undef TCP_GATE_UP_MIXED
#undef TCP_GATE_MIXED
#undef TCP_UP_MIXED
#undef TCP_DOWN_MIXED

template <int intermediate, bool mixed>
__device__ __forceinline__ void
tcp_down_launch(const struct exl3_tcp_record &record, unsigned char *arena)
{
	const struct tcp_parts parts = tcp_parts_of(record, arena);
	const int32_t *sorted = parts.table + 4;
	/* a part's sorted positions: the first half's come first (tcp_route's halves) */
	const int64_t first =
		record.part == EXL3_TCP_SECOND_HALF ? parts.table[3] : 0;
	const int64_t tasks = record.part == EXL3_TCP_FIRST_HALF ?
				      parts.table[3] :
				      parts.table[0];
	for (int64_t position = first + (int64_t)blockIdx.x * TCP_DOWN_WARPS + (threadIdx.x >> 5); position < tasks;
	     position += (int64_t)gridDim.x * TCP_DOWN_WARPS) {
		const int expert = (int)tcp_pick(record, arena, sorted[position]).expert;
		const uint64_t gate = tcp_matrix(record, arena, expert, 0), up = tcp_matrix(record, arena, expert, 1);
		const uint64_t down = tcp_matrix(record, arena, expert, 2);
		const uint32_t width = mixed ? tcp_width(record, expert) : (uint32_t)intermediate;
		tcp_down_row<intermediate, mixed>(parts.gate_result + position * intermediate,
						  parts.up_result + position * intermediate,
						  (const __half *)(arena + tcp_svh(record, 0, gate, width)),
						  (const __half *)(arena + tcp_svh(record, 1, up, width)),
						  (const __half *)(arena + tcp_suh(record, 2, down, width)),
						  record.activation_limit, parts.unscale + position,
						  width != intermediate);
	}
}
extern "C" __global__ void __launch_bounds__(EXL3_TCP_DOWN_THREADS)
	exl3_tcp_down_i640(struct exl3_tcp_record record, unsigned char *arena)
{
	tcp_down_launch<640, false>(record, arena);
}
extern "C" __global__ void __launch_bounds__(EXL3_TCP_DOWN_THREADS)
	exl3_tcp_down_i512(struct exl3_tcp_record record, unsigned char *arena)
{
	tcp_down_launch<512, false>(record, arena);
}
extern "C" __global__ void __launch_bounds__(EXL3_TCP_DOWN_THREADS)
	exl3_tcp_down_m640(struct exl3_tcp_record record, unsigned char *arena)
{
	tcp_down_launch<640, true>(record, arena);
}

extern "C" __global__ void __launch_bounds__(EXL3_TCP_COMBINE_THREADS)
	exl3_tcp_combine(struct exl3_tcp_record record, unsigned char *arena)
{
	const struct tcp_parts parts = tcp_parts_of(record, arena);
	const int32_t *position =
		parts.table +
		tcp_position_offset(parts.pairs, EXL3_TCP_ROUTE_EXPERTS);
	/* a part's rows: the first half_rows, or the rest */
	const int64_t first =
		record.part == EXL3_TCP_SECOND_HALF ? record.half_rows : 0;
	const int64_t end = record.part == EXL3_TCP_FIRST_HALF ?
				    record.half_rows :
				    record.row_count;
	tcp_combine_task<TC_MOE_HIDDEN>(
		parts.down_result, end - first, (int)PAIRS_PER_ROW,
		[&](int64_t row, int route, int *at, const __half **svh, double *weight) {
			const int64_t pair = (first + row) * PAIRS_PER_ROW + route;
			*at = position[pair];
			if (*at < 0)
				return false;
			const struct router_pick &pick =
				tcp_pick(record, arena, pair);
			*svh = (const __half *)(arena + tcp_svh(record, 2,
								tcp_matrix(record, arena, (int)pick.expert, 2),
								tcp_width(record, (int)pick.expert)));
			*weight = pick.route_weight;
			return true;
		},
		(float *)(arena + record.output_offset) + first * TC_MOE_HIDDEN);
}
