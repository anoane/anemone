/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Single-writer mapped-host publication. Only the aligned 32-bit commit word is a synchronization primitive.
 * The producer writes sequence words/payload before a system release of a fresh per-slot generation. A reader
 * acquires that exact generation before loading the sequence words. Generations advance once per actual slot
 * use, modulo uint32, with acknowledgement before reuse; deriving them from a skipped engine sequence is unsafe.
 * No participant resets an active channel. Full engine-sequence wrap requires a drained, reinitialized channel. */
#ifndef KERNEL_MAILBOX_SIGNAL_CUH
#define KERNEL_MAILBOX_SIGNAL_CUH

#include <stdint.h>

#define MAILBOX_SEQUENCE_HIGH_SHIFT 32u
#define MAILBOX_SEQUENCE_WORDS 2u

__device__ static __forceinline__ uint32_t mailbox_acquire32(const void *address)
{
	uint32_t value;
	asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
		     : "=r"(value)
		     : "l"(address)
		     : "memory");
	return value;
}

__device__ static __forceinline__ void mailbox_release32(void *address,
							 uint32_t value)
{
	asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(address),
		     "r"(value)
		     : "memory");
}

/* Data words, never a 64-bit mapped-host atomic. Read only after the caller acquired the fresh commit word. */
__device__ static __forceinline__ void mailbox_sequence_store(void *address,
							      uint64_t sequence)
{
	uint32_t *const words = (uint32_t *)address;
	words[0] = (uint32_t)sequence;
	words[MAILBOX_SEQUENCE_WORDS - 1u] =
		(uint32_t)(sequence >> MAILBOX_SEQUENCE_HIGH_SHIFT);
}

__device__ static __forceinline__ uint64_t
mailbox_sequence_load(const void *address)
{
	const uint32_t *const words = (const uint32_t *)address;
	uint32_t low, high;
	/* Reload mapped host words through the system path; no cached payload observation substitutes for acquire. */
	asm volatile("ld.global.cv.v2.u32 {%0, %1}, [%2];"
		     : "=r"(low), "=r"(high)
		     : "l"(words)
		     : "memory");
	return (uint64_t)low | ((uint64_t)high << MAILBOX_SEQUENCE_HIGH_SHIFT);
}

#endif
