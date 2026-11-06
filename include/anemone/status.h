/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * status.h: the one way every Anemone function reports failure. Host C11 only: no kernel file includes it.
 *
 * Contract: a function that can fail returns an result_status and gives its results through output parameters.
 * On failure it leaves every scalar output untouched; the contents of a caller's buffer it was filling are
 * unspecified and must not be used. STATUS_OK is zero. Nothing in the engine calls exit() or abort() on
 * bad input: bad input is a status. (ASSERT is for the engine's own invariants and exists only in debug
 * and thread builds.)
 */
#ifndef STATUS_H
#define STATUS_H

#ifdef __cplusplus
#error "anemone/status.h is host C11: no kernel file includes it"
#endif

/* A status must be looked at: the build refuses a call that drops one. Written before the return type,
 * `MUST_CHECK result_status name(...);`, and for a table member
 * `MUST_CHECK result_status (*read)(...);`. Every function and member that returns a status carries it. */
#define MUST_CHECK __attribute__((warn_unused_result))

typedef enum result_status {
	STATUS_OK = 0,
	STATUS_INVALID_ARGUMENT, /* the caller passed something no caller may pass */
	STATUS_OUT_OF_MEMORY, /* host or card memory could not be obtained */
	STATUS_INPUT_OUTPUT_ERROR, /* a drive refused, failed or returned fewer bytes than asked */
	STATUS_CORRUPT_PACK, /* a pack file contradicts itself or its digests */
	STATUS_DEVICE_ERROR, /* a card or its driver reported an error */
	STATUS_TIMED_OUT, /* a bounded wait ran out */
	STATUS_UNSUPPORTED, /* a valid request this build or this machine cannot serve */
	STATUS_BUSY_VALUE, /* a bounded queue or slot is full now; the same call may succeed later */
	STATUS_INJECTED_FAULT, /* a test armed a fault point (never in release builds) */
	STATUS_COUNT /* number of statuses; not a status */
} result_status;

/* A short English phrase for a status; never NULL, also for values outside the enumeration. */
const char *status_text(result_status status);

#endif
