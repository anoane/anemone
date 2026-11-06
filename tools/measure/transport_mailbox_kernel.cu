/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Device-only qualification kernel. All parties use aligned single-writer lines; no remote RMW atomics. */
#include "transport_mailbox_records.h"

__device__ __forceinline__ uint64_t transport_time(void)
{
    uint64_t result;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(result));
    return result;
}

__device__ __forceinline__ uint64_t transport_acquire(const uint64_t *address)
{
    uint64_t result;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(result) : "l"(address) : "memory");
    return result;
}

__device__ __forceinline__ void transport_release(uint64_t *address, uint64_t value)
{
    asm volatile("st.release.sys.global.u64 [%0], %1;" : : "l"(address), "l"(value) : "memory");
}

__device__ __forceinline__ uint4 transport_load(const uint4 *address)
{
    uint4 result;
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.w) : "l"(address) : "memory");
    return result;
}

__device__ __forceinline__ void transport_store(uint4 *address, uint4 value)
{
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" :
                 : "l"(address), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w) : "memory");
}

__device__ uint32_t transport_wait(uint64_t *line, uint64_t wanted, uint64_t deadline, uint64_t *seen)
{
    const uint64_t started = transport_time();
    do {
        *seen = transport_acquire(line);
        if (*seen == wanted) {
            return TRANSPORT_FAILURE_NONE;
        }
        if (*seen > wanted) {
            return TRANSPORT_FAILURE_AHEAD;
        }
    } while (transport_time() - started < deadline);
    return TRANSPORT_FAILURE_LATE;
}

template<uint32_t fault_mode, bool validate_payload, bool cache_published>
__device__ void transport_run(struct transport_arguments arguments)
{
    __shared__ uint32_t failure;
    __shared__ uint64_t seen;
    __shared__ uint32_t mismatched_vectors;
    __shared__ uint64_t last_published[TRANSPORT_PARTICIPANTS];
    struct transport_status *status = (struct transport_status *)arguments.status;
    const uint4 *input = (const uint4 *)arguments.input;
    uint4 *output = (uint4 *)arguments.output;
    uint64_t mismatches = 0;
    uint32_t completed = 0;
    uint64_t sequence = arguments.sequence_base;
    unsigned char *control = (unsigned char *)arguments.control;
    if (threadIdx.x == 0) {
	    transport_release(
		    (uint64_t *)(control + TRANSPORT_READY_OFFSET +
				 arguments.role * EXCHANGE_LINE_BYTES),
		    sequence + 1u);
	    failure = transport_wait(
		    (uint64_t *)(control + TRANSPORT_READY_OFFSET +
				 (1u - arguments.role) * EXCHANGE_LINE_BYTES),
		    sequence + 1u, arguments.deadline_nanoseconds, &seen);
    }
    __syncthreads();
    if (failure != TRANSPORT_FAILURE_NONE) {
        if (threadIdx.x == 0) {
            status->failure = failure;
        }
        return;
    }
    const uint64_t began = transport_time();
    if (arguments.role == 0 && fault_mode == TRANSPORT_FAULT_MISSING_SENDER) {
        return;
    }
    if (arguments.role == 0 && cache_published && threadIdx.x == 0) {
        for (uint32_t parity = 0; parity < TRANSPORT_PARTICIPANTS; parity++) {
            last_published[parity] = transport_acquire((uint64_t *)(arguments.mailbox +
                                                    parity * arguments.parity_stride));
        }
    }
    __syncthreads();
    /* The last message is a zero-byte drain; it must still publish, validate and acknowledge its trailer. */
    for (uint32_t round = 1; round <= arguments.round_count + 1; round++) {
        /* All warps must finish reading the preceding failure value before the polling warp replaces it. */
        __syncthreads();
        sequence = arguments.sequence_base + round;
        const uint32_t bytes = round <= arguments.round_count ? arguments.byte_count : 0;
	const uint32_t vectors = bytes / EXCHANGE_WORD_BYTES;
	unsigned char *slot = (unsigned char *)arguments.mailbox + (sequence & 1u) * arguments.parity_stride;
        uint64_t *flag = (uint64_t *)slot;
        uint64_t *credit = (uint64_t *)(arguments.credit + (sequence & 1u) * arguments.credit_parity_stride);
        uint4 *payload = (uint4 *)(slot + TRANSPORT_PAYLOAD_OFFSET);
	struct exchange_trailer *trailer =
		(struct exchange_trailer *)(slot + TRANSPORT_PAYLOAD_OFFSET +
					    bytes);
	if (threadIdx.x == 0) {
            const uint64_t wanted = arguments.role == 0 ?
                (cache_published ? last_published[sequence & 1u] : transport_acquire(flag)) : sequence;
            failure = transport_wait(arguments.role == 0 ? credit : flag, wanted,
                                     arguments.deadline_nanoseconds, &seen);
        }
        __syncthreads();
        if (failure != TRANSPORT_FAILURE_NONE) {
            break;
        }
        /* Each process maps these physical pages at its own virtual address. */
        asm volatile("fence.proxy.alias;" : : : "memory");
        __syncthreads();
        if (arguments.role != 0 && threadIdx.x == 0) {
		const volatile struct exchange_trailer *published = trailer;
		if (published->sequence != sequence ||
		    published->byte_count != bytes ||
		    published->site != TRANSPORT_SITE ||
		    published->card != arguments.sender) {
			failure = TRANSPORT_FAILURE_TRAILER;
		}
        }
        __syncthreads();
        if (failure != TRANSPORT_FAILURE_NONE) {
            break;
        }
        if (validate_payload && threadIdx.x == 0) {
            mismatched_vectors = 0;
        }
        __syncthreads();
        uint32_t bad = 0;
        for (uint32_t vector = threadIdx.x; vector < vectors; vector += blockDim.x) {
            const uint32_t mask = (uint32_t)sequence;
            if (arguments.role == 0) {
                uint4 value = transport_load(input + vector);
                value.x ^= mask;
                value.y ^= mask;
                value.z ^= mask;
                value.w ^= mask;
                if (round == TRANSPORT_CORRUPT_SEQUENCE && vector == 0 &&
                    fault_mode == TRANSPORT_FAULT_PAYLOAD) {
                    value.x = ~value.x;
                }
                transport_store(payload + vector, value);
            } else {
                const uint4 value = transport_load(payload + vector);
                if (validate_payload) {
                    const uint32_t expected = (vector * TRANSPORT_PATTERN_MULTIPLIER) ^ TRANSPORT_PATTERN_SEED ^ mask;
                    bad += value.x != expected || value.y != expected || value.z != expected || value.w != expected;
                }
                transport_store(output + vector, value);
            }
        }
        if (validate_payload && bad != 0) {
            atomicAdd(&mismatched_vectors, bad); /* Block-local reduction only; no shared-host or peer atomics. */
        }
        /* Every writer fences its own writes before the publishing thread releases the line. */
        __threadfence_system();
        asm volatile("fence.proxy.alias;" : : : "memory");
        __syncthreads();
        if (threadIdx.x == 0) {
            if (validate_payload) {
                mismatches += mismatched_vectors;
            }
            if (validate_payload && mismatched_vectors != 0) {
                failure = TRANSPORT_FAILURE_PAYLOAD;
            } else if (arguments.role == 0) {
                trailer->sequence = sequence + (fault_mode == TRANSPORT_FAULT_TRAILER_SEQUENCE &&
                                                round == TRANSPORT_CORRUPT_SEQUENCE);
                trailer->byte_count = bytes + (fault_mode == TRANSPORT_FAULT_TRAILER_SIZE &&
                                               round == TRANSPORT_CORRUPT_SEQUENCE);
                trailer->site = TRANSPORT_SITE;
                trailer->card = (uint16_t)arguments.sender;
                const uint64_t published = sequence + (fault_mode == TRANSPORT_FAULT_AHEAD &&
                                                        round == TRANSPORT_CORRUPT_SEQUENCE);
                transport_release(flag, published);
                if (cache_published) {
                    last_published[sequence & 1u] = published;
                }
            } else {
                transport_release(credit, sequence);
            }
        }
        __syncthreads();
        if (failure != TRANSPORT_FAILURE_NONE) {
            break;
        }
        completed = round;
    }
    __syncthreads();
    /* Sender completion includes the receiver's final drain acknowledgement, not merely stores being issued. */
    if (threadIdx.x == 0) {
        uint32_t final_failure = failure;
        if (arguments.role == 0 && final_failure == TRANSPORT_FAILURE_NONE) {
            final_failure = transport_wait((uint64_t *)(arguments.credit +
                                           (sequence & 1u) * arguments.credit_parity_stride), sequence,
                                           arguments.deadline_nanoseconds, &seen);
        }
        status->sequence = sequence;
        status->seen = seen;
        status->mismatched_vectors = mismatches;
        status->elapsed_nanoseconds = transport_time() - began;
        status->completed = completed;
        status->failure = final_failure;
    }
}

/* Diagnostics and fault branches are absent from the timed instantiations. */
extern "C" __global__ void transport_mailbox(struct transport_arguments arguments)
{
    transport_run<TRANSPORT_FAULT_NONE, false, false>(arguments);
}
extern "C" __global__ void transport_mailbox_cached(struct transport_arguments arguments)
{
    transport_run<TRANSPORT_FAULT_NONE, false, true>(arguments);
}
extern "C" __global__ void transport_mailbox_validate(struct transport_arguments arguments)
{
    transport_run<TRANSPORT_FAULT_NONE, true, false>(arguments);
}
extern "C" __global__ void transport_mailbox_cached_validate(struct transport_arguments arguments)
{
    transport_run<TRANSPORT_FAULT_NONE, true, true>(arguments);
}
extern "C" __global__ void transport_mailbox_fault(struct transport_arguments arguments)
{
    switch (arguments.fault) {
        case TRANSPORT_FAULT_PAYLOAD: transport_run<TRANSPORT_FAULT_PAYLOAD, true, true>(arguments); break;
        case TRANSPORT_FAULT_TRAILER_SEQUENCE:
            transport_run<TRANSPORT_FAULT_TRAILER_SEQUENCE, true, true>(arguments); break;
        case TRANSPORT_FAULT_TRAILER_SIZE: transport_run<TRANSPORT_FAULT_TRAILER_SIZE, true, true>(arguments); break;
        case TRANSPORT_FAULT_AHEAD: transport_run<TRANSPORT_FAULT_AHEAD, true, true>(arguments); break;
        case TRANSPORT_FAULT_MISSING_SENDER:
            transport_run<TRANSPORT_FAULT_MISSING_SENDER, true, true>(arguments); break;
        default: break;
    }
}
