/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ordered_collective_records.h: small peer collectives, C11/CUDA records.
 *
 * Each rank owns two slots. A slot starts with three separately owned, system-scope publication lines, followed by
 * input, output and scratch arrays of element_count float32 values. peer_table_offset holds rank_count pairs of
 * {arena_address, region_offset, host_address, route, reserved_zero}; self uses its supplied GPU arena. A remote
 * direct route addresses that rank's VRAM; a host route addresses its mapped, registered host mirror. host_address
 * on the local rank's entry is its own mapped mirror, or zero when no rank needs it. It publishes matching inputs,
 * two-shot scratch slices and phase signals there inside the same kernel. The backend writes addresses once.
 * A failed direct grant can therefore select a host route without changing the operation or arithmetic.
 * All addresses and both slots remain valid until every
 * participant has drained. No participant may overlap another collective using these slots.
 *
 * All ranks submit identical positive, strictly increasing sequences, rank_count, element_count, array offsets,
 * stride and operation. step_offset names a step block. Inputs precede this launch on each rank's stream;
 * outputs follow it. Parity is sequence & 1. All ranks reach the kernel independently. One block publishes its
 * input, waits for all inputs, and sums +0, rank 0, rank 1, ... with one rounded FP32 add per rank. It never forms
 * rank partial sums. The two-shot form reduces disjoint contiguous slices, publishes them, and gathers each slice.
 * Reduce-scatter writes its own slice at its full-array output indices; gather copies those indexed slices from
 * every rank's input array. These standalone operations therefore compose with the same FP32 bits as two-shot:
 * use a fresh sequence of the same parity for gather and permute its input/output offsets to read the scatter's
 * output. A second operation must never reuse a sequence: the previous ready lines would not order its data.
 *
 * The final all-rank read-complete barrier releases the input and scratch before any rank returns. Consequently a
 * later producer in that stream can safely reuse a slot. Signals are monotonic and are never reset while active.
 * Each phase/slot has a naturally aligned 32-bit commit generation and two 32-bit full-sequence data words. Only
 * that rank writes the line. Generations advance once per actual phase/slot use, modulo uint32, independently of
 * skipped engine sequences. Acquire of the next commit precedes full-sequence validation and payload reads. The
 * prior commit always differs from the next, including at uint32 wrap. All ranks use the same phases in order.
 * No mapped-host RMW or 64-bit atomic is used. Full uint64 engine-sequence wrap/zero is refused; after cancellation
 * or failure every participant must drain/close before a fresh zeroed preparation. Waits expire after
 * deadline_cycles and trap, so stale payload is never consumed.
 *
 * This is exact for finite FP32 inputs whose ordered sum is not NaN. A TP decomposition that changes the original
 * product's add tree needs a separate exactness proof. EP must preserve individual slots before its ordered sum;
 * summing rank-local expert partials is not equivalent.
 */
#ifndef ORDERED_COLLECTIVE_RECORDS_H
#define ORDERED_COLLECTIVE_RECORDS_H

#include <stdint.h>
#include "anemone/step_records.h"

#define KERNEL_FILE_ORDERED_COLLECTIVE "kernel_ordered_collective"
#define KERNEL_ORDERED_ALL_REDUCE "ordered_all_reduce"
#define KERNEL_ORDERED_TWO_SHOT "ordered_two_shot"
#define KERNEL_ORDERED_REDUCE_SCATTER "ordered_reduce_scatter"
#define KERNEL_ORDERED_GATHER "ordered_gather"
#define ORDERED_COLLECTIVE_MAXIMUM_RANKS 16u
#define ORDERED_COLLECTIVE_THREADS 256u
#define ORDERED_COLLECTIVE_LINE_BYTES 128ull
#define ORDERED_COLLECTIVE_PHASES 3u
#define ORDERED_COLLECTIVE_SIGNAL_BYTES \
	(ORDERED_COLLECTIVE_PHASES * ORDERED_COLLECTIVE_LINE_BYTES)
#define ORDERED_COLLECTIVE_SLOTS 2u
#define ORDERED_COLLECTIVE_PEER_BYTES 32u
#define ORDERED_COLLECTIVE_ARGUMENT_BYTES 80u
#define ORDERED_COLLECTIVE_MAXIMUM_ELEMENTS 262144u
#define ORDERED_COLLECTIVE_SEQUENCE_OFFSET 8u
#define ORDERED_COLLECTIVE_DIRECT_ROUTE 0u
#define ORDERED_COLLECTIVE_HOST_ROUTE 1u

enum ordered_collective_operation {
	ORDERED_ALL_REDUCE,
	ORDERED_TWO_SHOT,
	ORDERED_REDUCE_SCATTER,
	ORDERED_GATHER,
	ORDERED_COLLECTIVE_OPERATIONS
};

struct ordered_collective_peer {
	uint64_t arena_address;
	uint64_t region_offset;
	uint64_t host_address;
	uint32_t route;
	uint32_t reserved_zero;
};

struct ordered_collective_arguments {
	uint64_t peer_table_offset;
	uint64_t region_offset;
	uint64_t input_offset;
	uint64_t output_offset;
	uint64_t scratch_offset;
	uint64_t step_offset;
	uint64_t parity_stride;
	uint64_t deadline_cycles;
	uint32_t element_count;
	uint32_t rank_count;
	uint32_t rank;
	uint32_t reserved_zero;
};

STEP_RECORD_HAS(ordered_collective_peer, ORDERED_COLLECTIVE_PEER_BYTES);
STEP_RECORD_HAS(ordered_collective_arguments,
		ORDERED_COLLECTIVE_ARGUMENT_BYTES);

#endif
