/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* exl3_tc_records.h: the routed experts of one EXL3 layer on the tensor cores, from the router's job table, one TP
 * rank's intermediate slice: the native rewrite of GNmac's MoE (Morrowmake's `moe_out_v2_try`, Curtis's tc-moe shape,
 * gnmac_port/kernels/kernel_exl3_tc_moe.cuh), selected by --exl3_tc_moe (GNmac numerics, default off). Every result
 * is the original's arithmetic: m16n8k16 FP16 MMAs on exactly decoded weights against a hi/lo FP16 split of the
 * scaled activations, FP64 input, activation and combine transforms, fixed order, a row never depending on another.
 * Module EXL3_TC_KERNEL_FILE (sm_80 only). Five launches a layer (exl3_tc.c): operands (block 0 counts the
 * jobs and lays out each pair's down transform and weight), gate/up products, activation, down products, combine. */
#ifndef EXL3_TC_RECORDS_H
#define EXL3_TC_RECORDS_H
#include <stdbool.h>
#include <stdint.h>

#define EXL3_TC_KERNEL_FILE "kernel_exl3_tc"
#define EXL3_TC_LAUNCHES 5u
#define EXL3_TC_HIDDEN 5120u
#define EXL3_TC_PAIRS_PER_ROW 6u
#define EXL3_TC_MOST_PAIRS \
	2688u /* routed pairs of one call (a 448-row tail program's) */
#define EXL3_TC_MOST_ROWS \
	64u /* the engine's calls: decode-sized steps (GNmac: 1-64 rows) */
#define EXL3_TC_LONG_PAIRS \
	160u /* from here: long gate/up slices, one down slice */
#define EXL3_TC_GATE_UP_SLICES 16u /* short; the long path's 8 fit its space */
#define EXL3_TC_DOWN_SLICES 2u
#define EXL3_TC_JOB_ROWS 8u /* rows of one job (LINEAR_MAXIMUM_ROWS) */
#define EXL3_TC_OPERAND_THREADS 640u
#define EXL3_TC_PRODUCT_THREADS 256u
#define EXL3_TC_PRODUCT_BLOCKS \
	148u /* CMP 170HX: 74 multiprocessors x 2 resident product blocks */
#define EXL3_TC_COMBINE_THREADS 192u /* a warp a route */
#define EXL3_TC_MOST_BLOCKS 65535u
#define EXL3_TC_ACTIVATE_GRID \
	592u /* activation blocks: one wave (74 x 8); they loop */
#define EXL3_TC_STASH_BYTES \
	4096u /* a paired item's first-slice sums: 8 tiles x 8 values x 16 lanes, FP32 */

/* One call. Matrices (the jobs' offsets) are EXL3: trellis [input/16][output/16][8 bits_x2] uint16, then FP16
 * suh[input], svh[output]; gate and up 5120 x intermediate, down intermediate x 5120, each 16-byte aligned. */
struct exl3_tc_record {
	uint64_t jobs_offset; /* read: struct expert_job_record [job_capacity], ascending experts, empty ones last */
	uint64_t hidden_offset;    /* read: bfloat16 [row_count][5120] */
	uint64_t workspace_offset; /* scratch: exl3_tc_workspace_bytes(pair_count, intermediate), 256-aligned */
	uint64_t output_offset;    /* written: float32 [row_count][5120], each row's route-weighted expert sum */
	uint32_t row_count;
	uint32_t job_capacity;
	uint32_t intermediate;     /* 640 or 512 */
	uint32_t bits_x2;          /* 6 (K3) or 4 (K2) */
	float activation_limit;    /* the SwiGLU limit; 0: none */
	uint32_t pass;             /* the products launches: 0 gate and up, 1 down; the others 0 */
	uint32_t rotation; /* 0, or --exl3_expert_rotation's mixed widths (exl3_tc_width) at 640 */
	/* 0: the combine writes each row's sum FP32 at output_offset; 1: it adds the sum to the shared expert's
	 * bfloat16 rows at shared_offset ([row_count][5120], read and written in place) and rounds to bfloat16 (the
	 * BF16 ring's input; tp2_join's BFLOAT16_SUM_TO_BFLOAT16 arithmetic: the bits of the FP32 sums and that
	 * join) */
	uint32_t shared_sum;
	uint64_t shared_offset;
};

/* The layout functions serve the host's preparation and the kernels alike. */
#ifdef __CUDACC__
#define EXL3_TC_LAYOUT __host__ __device__
#else
#define EXL3_TC_LAYOUT
#endif

/* --exl3_expert_rotation (default 0): the routed experts' 2,304 columns cut in the chunks 512, 640, 512 and 640 (the
 * 128-column split of exl3_layout, chunk c rank c's without rotation), rotated per expert e of target layer L:
 *   1 (Morrow's rotation, a5693f7): rank r holds chunk (r + L + e) % 4, so every rank holds the 640-column chunk of
 *     half of a layer's experts; every rank's experts grow to 576 columns on average (rank 0, which also holds the
 *     head, the embedding and the draft, then needs ~5 GB more);
 *   2 (three-way): rank 0 keeps chunk 0 (512 columns), rank r of 1-3 holds chunk 1 + (r - 1 + L + e) % 3, so ranks 1-3
 *     hold the 640-column chunk of two thirds of the experts (597 columns on average, against 640 and 512).
 * A record's rotation then names its narrow experts (expert e intermediate - 128 wide): 1 or 2 where (e + rotation)
 * is even, 3 to 5 where (e + rotation) % 3 is 0 (exl3_tc_rotation_of). Its workspace and operand rows keep the
 * 640 layout; each routed chunk's product is bitwise the base kernels' at its width, the rank's partial groups other
 * chunks (another order of the TP sum). */
#define EXL3_TC_NARROWER 128u
#define EXL3_TC_MIXED_WIDTH 640u
#define EXL3_TC_ROTATION_FOUR_WAY 1u
#define EXL3_TC_ROTATION_THREE_WAY 2u
#define EXL3_TC_THREE_WAY_FIRST \
	3u /* a record's three-way rotation values: 3, 4, 5 */
#define EXL3_TC_THREE_WAY_RANKS 3u
#define EXL3_TC_FOUR_WAY_RANKS 4u
EXL3_TC_LAYOUT static inline uint32_t
exl3_tc_width(uint32_t intermediate, uint32_t rotation, uint32_t expert)
{
	const bool narrow =
		rotation == 0u ?
			false :
		rotation < EXL3_TC_THREE_WAY_FIRST ?
			((expert + rotation) & 1u) == 0u :
			(expert + rotation) % EXL3_TC_THREE_WAY_RANKS == 0u;
	return narrow ? intermediate - EXL3_TC_NARROWER : intermediate;
}
/* The chunk of expert `expert` rank `rank` holds in target layer `layer` under rotation `mode` (the rank's own
 * without), and the rotation of the rank's records in that layer (0: every expert its chunk's width). */
EXL3_TC_LAYOUT static inline uint32_t
exl3_tc_chunk_of(uint32_t mode, uint32_t rank, uint32_t layer, uint32_t expert)
{
	return mode == EXL3_TC_ROTATION_FOUR_WAY ?
		       (rank + layer + expert) % EXL3_TC_FOUR_WAY_RANKS :
	       mode == EXL3_TC_ROTATION_THREE_WAY && rank != 0u ?
		       1u + (rank - 1u + layer + expert) %
				       EXL3_TC_THREE_WAY_RANKS :
		       rank;
}
EXL3_TC_LAYOUT static inline uint32_t
exl3_tc_rotation_of(uint32_t mode, uint32_t rank, uint32_t layer)
{
	/* four-way: narrow (an even chunk) where (rank + layer + expert) is even; three-way, ranks 1-3: narrow (chunk 2)
	 * where (rank - 1 + layer + expert) % 3 is 1, i.e. (expert + rank + layer + 1) % 3 is 0 */
	return mode == EXL3_TC_ROTATION_FOUR_WAY ?
		       (((rank + layer) & 1u) != 0u ? 1u : 2u) :
	       mode == EXL3_TC_ROTATION_THREE_WAY && rank != 0u ?
		       EXL3_TC_THREE_WAY_FIRST +
			       (rank + layer + 1u) % EXL3_TC_THREE_WAY_RANKS :
		       0u;
}

/* The workspace of pair_count = row_count x 6 pairs: FP32 gate, up and down slice partials; the FP16 hi/lo gate/up and
 * down operands; their 2^-e; the job count; per pair its down svh offset and route weight. Offsets in bytes. */
EXL3_TC_LAYOUT static inline uint64_t exl3_tc_aligned(uint64_t bytes)
{
	return (bytes + 255u) / 256u * 256u;
}
enum exl3_tc_part {
	EXL3_TC_GATE,
	EXL3_TC_UP,
	EXL3_TC_DOWN,
	EXL3_TC_GATE_UP_HALVES,
	EXL3_TC_DOWN_HALVES,
	EXL3_TC_GATE_UP_UNSCALE,
	EXL3_TC_DOWN_UNSCALE,
	EXL3_TC_COUNT,
	EXL3_TC_BY_PAIR,
	EXL3_TC_PARTS
};
struct exl3_tc_by_pair { /* a pair's down svh (arena offset), route weight, and whether a job holds it */
	uint64_t svh_offset;
	float route_weight;
	uint32_t is_routed;
};
EXL3_TC_LAYOUT static inline uint64_t
exl3_tc_part_bytes(uint32_t part, uint64_t pair_count, uint64_t intermediate)
{
	switch (part) {
	case EXL3_TC_GATE:
	case EXL3_TC_UP:
		return EXL3_TC_GATE_UP_SLICES * pair_count * intermediate * 4u;
	case EXL3_TC_DOWN:
		return EXL3_TC_DOWN_SLICES * pair_count * EXL3_TC_HIDDEN * 4u;
	case EXL3_TC_GATE_UP_HALVES:
		return pair_count * 2u * 2u * EXL3_TC_HIDDEN * 2u;
	case EXL3_TC_DOWN_HALVES:
		return pair_count * 2u * intermediate * 2u;
	case EXL3_TC_GATE_UP_UNSCALE:
		return pair_count * 2u * EXL3_TC_GATE_UP_SLICES * 4u;
	case EXL3_TC_DOWN_UNSCALE:
		return pair_count * EXL3_TC_DOWN_SLICES * 4u;
	case EXL3_TC_COUNT:
		return 16u;
	case EXL3_TC_BY_PAIR:
		return pair_count * sizeof(struct exl3_tc_by_pair);
	default:
		return 0;
	}
}
EXL3_TC_LAYOUT static inline uint64_t
exl3_tc_part_offset(uint32_t part, uint64_t pair_count, uint64_t intermediate)
{
	uint64_t offset = 0;
	for (uint32_t before = 0; before < part; before++)
		offset += exl3_tc_aligned(
			exl3_tc_part_bytes(before, pair_count, intermediate));
	return offset;
}
EXL3_TC_LAYOUT static inline uint64_t
exl3_tc_workspace_bytes(uint64_t pair_count, uint64_t intermediate)
{
	return exl3_tc_part_offset(EXL3_TC_PARTS, pair_count, intermediate);
}

/* ---- the prefill form (kernel_exl3_tcp.cuh, Morrowmake's `moe_out_prefill_try`): calls of more than 64 rows ----
 * Six or eight launches a sequence of at most EXL3_TCP_CHUNK_ROWS rows: route (one block, from the router's
 * picks), gate/up operands, gate/up GEMMs (both at once up to EXL3_TCP_JOINT_ROWS rows, else gate then up),
 * down operands, down GEMM, combine (FP32 out). The pass field names the GEMM launch's projections. With half_rows (a
 * joint call's first half: split_overlap mode 7) the route gives an expert's two halves adjacent positions and the
 * gate/up tiles over both (each expert's weights decoded once a tile, as the joint call's), a retile behind the
 * gate/up GEMMs makes the down tiles (an expert of at most 32 pairs one over both halves, else each half's), the down
 * operands run over the whole call, then the down GEMM of the first part (the joint tiles and the first half's) and
 * the first half's combine, then the second half's tiles and combine (part): every row's arithmetic the whole
 * call's, so the first half's FFN sums may begin while the second half's down GEMM runs. */
#define EXL3_TCP_KERNEL_FILE "kernel_exl3_tcp"
#define EXL3_TCP_TILE_ROWS 64u
#define EXL3_TCP_CHUNK_ROWS 2048u
#define EXL3_TCP_JOINT_ROWS 1024u
#define EXL3_TCP_MOST_LAUNCHES \
	11u /* route, two gate/up operand and GEMM launches, retile, two parts */
#define EXL3_TCP_PART_LAUNCHES \
	3u /* half_rows' first part: down operands (the whole call's), down GEMM,
						combine */
#define EXL3_TCP_SECOND_PART_LAUNCHES \
	2u /* its second part: down GEMM, combine */
#define EXL3_TCP_EXPERTS 384u
#define EXL3_TCP_ROUTE_EXPERTS \
	(2u * EXL3_TCP_EXPERTS) /* the route's: an expert a half (half_rows) */
#define EXL3_TCP_ROUTE_THREADS 1024u
#define EXL3_TCP_OPERAND_THREADS \
	160u /* 5 warps: two 512-column spans of the hidden row each */
#define EXL3_TCP_DOWN_THREADS 64u
#define EXL3_TCP_GEMM_THREADS 256u
#define EXL3_TCP_COMBINE_THREADS 256u
enum exl3_tcp_pass {
	EXL3_TCP_GATE_AND_UP = 0,
	EXL3_TCP_GATE = 1,
	EXL3_TCP_UP = 2,
	EXL3_TCP_DOWN = 3
};
/* The rows a down operand, down GEMM or combine launch takes: all, or with half_rows the first half's or the rest. */
enum exl3_tcp_part {
	EXL3_TCP_WHOLE = 0,
	EXL3_TCP_FIRST_HALF = 1,
	EXL3_TCP_SECOND_HALF = 2
};
struct exl3_tcp_record {
	uint64_t picks_offset; /* read: struct router_pick [row_count][6] (expert, route weight) */
	uint64_t expert_offsets_offset; /* read: uint64 [384][3], each expert's gate, up and down matrix offsets */
	uint64_t hidden_offset;         /* read: bfloat16 [row_count][5120] */
	uint64_t workspace_offset; /* scratch: exl3_tcp_workspace_bytes(row_count, intermediate) */
	uint64_t output_offset;         /* written: float32 [row_count][5120] */
	uint32_t row_count; /* 1 to EXL3_TCP_CHUNK_ROWS */
	uint32_t intermediate;          /* 640 or 512 */
	uint32_t bits_x2;               /* 6 or 4 */
	uint32_t pass; /* the GEMM launches: enum exl3_tcp_pass; the others 0 */
	float activation_limit;
	uint32_t rotation; /* 0, or the mixed widths at 640 (exl3_tc_width) */
	uint32_t half_rows; /* 0, or 1 to row_count - 1: the first half's rows (the route's halves apart) */
	uint32_t part; /* a down operand, down GEMM or combine launch's rows: enum exl3_tcp_part; others 0 */
	uint64_t step_offset; /* with cone_layer: the step area of the call's rows (its block, then its rows) */
	uint32_t cone_layer; /* 0, or a decoder layer: rows step_records.h's STEP_ROW_CONE_SKIPS leaves alone
				 take no route (their sums stay zero) */
	uint32_t two_blocks; /* 0, or 1 (--exl3_tcp_two_blocks): the GEMMs' two-blocks forms (_b2), the same bits */
};
/* The route table's tiles at most (every route's: an expert a half). */
EXL3_TC_LAYOUT static inline uint64_t exl3_tcp_tile_capacity(uint64_t pairs)
{
	const uint64_t experts = EXL3_TCP_ROUTE_EXPERTS;
	const uint64_t bound = pairs / EXL3_TCP_TILE_ROWS +
			       (experts < pairs ? experts : pairs) + 1u;
	return bound < pairs ? bound : pairs;
}
/* FP32 words: the operand region (the gate/up split rows of the running launch, then the down results), gate results
 * (then the down split rows in place), up results, 2^-e of the running launch, the route table (int32). */
EXL3_TC_LAYOUT static inline uint64_t
exl3_tcp_sequence_words(uint64_t rows, uint64_t intermediate)
{
	const uint64_t pairs = rows * EXL3_TC_PAIRS_PER_ROW,
		       projections = rows <= EXL3_TCP_JOINT_ROWS ? 2u : 1u;
	const uint64_t table = 4u + (pairs + 3u) / 4u * 4u +
			       4u * exl3_tcp_tile_capacity(pairs) + pairs;
	return pairs * projections * EXL3_TC_HIDDEN +
	       pairs * 2u * intermediate +
	       (pairs * projections + 3u) / 4u * 4u + (table + 3u) / 4u * 4u;
}
EXL3_TC_LAYOUT static inline uint64_t
exl3_tcp_workspace_bytes(uint64_t rows, uint64_t intermediate)
{
	const uint64_t joint = rows < EXL3_TCP_JOINT_ROWS ? rows :
							    EXL3_TCP_JOINT_ROWS;
	const uint64_t first = exl3_tcp_sequence_words(rows, intermediate);
	const uint64_t second = exl3_tcp_sequence_words(joint, intermediate);
	return 4u * (first > second ? first : second);
}

#endif
